import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple
import h5py
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support

from algo import kerasutil
from algo import scoring
from algo import rl
from algo.goboard_fast import GameState
from algo.data.parallel_processor import GoDataProcessor
from algo.networks.alphago import alphago_model
from algo.agent.pg import PolicyAgent
from algo.agent.predict import load_prediction_agent
from algo.encoders.alphago import AlphaGoEncoder
from algo.rl.simulate import experience_simulation
from algo.rl import ValueAgent, load_experience
from algo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from algo.rl import load_value_agent
from algo.goboard_fast import GameState

def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return rl.load_q_agent(h5file)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_soft_device_placement(True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    else:
        return None

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename)
    agent1.set_temperature(temperature)
    agent2 = load_agent(agent2_filename)
    agent2.set_temperature(temperature)

    collector1 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
        else:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    experience = rl.combine_experience([collector1])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)



def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, board_size, num_workers, temperature):
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                temperature,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = rl.load_experience(expf)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = rl.load_experience(expf)
        combined_buffer = rl.combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)

    learning_agent = load_agent(learning_agent)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = rl.load_experience(expf)
    learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)

    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)



def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        )
    )
    worker.start()
    worker.join()


def play_games(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)

    agent1_fname, agent2_fname, num_games, board_size, gpu_frac, temperature = args

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent1.set_temperature(temperature)
    agent2 = load_agent(agent2_fname)
    agent2.set_temperature(temperature)

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size,
             temperature):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac,
            temperature,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Reference: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl-agent', '-sl', default="./alphago_agents/ag_sl.h5")
    parser.add_argument('--rl-agent', '-rl', default="./alphago_agents/ag_rl.h5")
    parser.add_argument('--mcts-agent', '-mc', default="./alphago_agents/ag_mcts.h5")
    parser.add_argument('--val-agent', '-v', default="./alphago_agents/ag_sl_val.h5")
    parser.add_argument('--sample-games', '-s', type=int, default=100)
    parser.add_argument('--self-play-games', '-g', type=int, default=100)
    parser.add_argument('--work-dir', '-d', default="./alphago_training/")
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--board-size', '-b', type=int, default=19)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', '--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l', default="./alphago_training/ag_log.txt")

    args = parser.parse_args()
    pwd = args.work_dir
    samp = 100
    epo = 1



    # tag::e2e_processor[]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data_dir = pwd + timestr

    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = AlphaGoEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name(), data_directory=data_dir)

    X, y = processor.load_go_data(num_samples=samp)
    # end::e2e_processor[]

    # tag::e2e_model[]
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    try:
        with tf.device('/device:GPU:0'):
            model.add(Dense(nb_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

            vals = model.fit(X, y, batch_size=128, epochs=epo, verbose=1)
            # end::e2e_model[]

            # tag::e2e_agent[]
            deep_learning_bot = DeepLearningAgent(model, encoder)
            deep_learning_bot.serialize(h5py.File(model_h5filename, "w"))
            # end::e2e_agent[]

            # tag::e2e_load_agent[]
            model_file = h5py.File(model_h5filename, "r")
            bot_from_file = load_prediction_agent(model_file)

            web_app = get_web_app({'predict': bot_from_file})
            web_app.run()
            # end::e2e_load_agent[]
    except RuntimeError as e:
        print(e)





    logf = open(args.log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting from %s at %s\n' % (
        args.agent, datetime.datetime.now()))

    temp_decay = 0.98
    min_temp = 0.01
    temperature = args.temperature

    learning_agent = args.rl_agent
    reference_agent = args.rl_agent
    experience_file = os.path.join(args.work_dir, 'exp_temp.hdf5')
    tmp_agent = os.path.join(args.work_dir, 'agent_temp.hdf5')
    working_agent = os.path.join(args.work_dir, 'agent_cur.hdf5')
    total_games = 0
    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        generate_experience(
            learning_agent, reference_agent,
            experience_file,
            num_games=args.games_per_batch,
            board_size=args.board_size,
            num_workers=args.num_workers,
            temperature=temperature)
        train_on_experience(
            learning_agent, tmp_agent, experience_file,
            lr=args.lr, batch_size=args.bs)
        total_games += args.games_per_batch
        wins = evaluate(
            learning_agent, reference_agent,
            num_games=480,
            num_workers=args.num_workers,
            board_size=args.board_size,
            temperature=temperature)
        print('Won %d / 480 games (%.3f)' % (
            wins, float(wins) / 480.0))
        logf.write('Won %d / 480 games (%.3f)\n' % (
            wins, float(wins) / 480.0))
        shutil.copy(tmp_agent, working_agent)
        learning_agent = working_agent
        if wins >= 262:
            next_filename = os.path.join(
                args.work_dir,
                'agent_%08d.hdf5' % (total_games,))
            shutil.move(tmp_agent, next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n' % next_filename)
            temperature = max(min_temp, temp_decay * temperature)
            logf.write('New temperature is %f\n' % temperature)
        else:
            print('Keep learning\n')
        logf.flush()


if __name__ == '__main__':
    main()

