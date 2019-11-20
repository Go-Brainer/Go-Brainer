# package imports
import multiprocessing
from multiprocessing import freeze_support
import os
import time
import h5py
import tensorflow as tf
import random
import numpy as np
from keras.optimizers import Adam, Adadelta, SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.callbacks import ModelCheckpoint
from datetime import datetime as dt
from copy import deepcopy

# module imports
from gbgo.agent import DeepLearningAgent, load_prediction_agent, load_policy_agent, GBAgent, PolicyAgent
from gbgo.encoders.sevenplane import SevenPlaneEncoder
from gbgo.networks import large
from gbgo.data.parallel_processor import GoDataProcessor
from gbgo.rl.simulate import experience_simulation
from gbgo.rl import ValueAgent, load_experience, load_value_agent
from gbgo.goboard_fast import GameState, Player, Point
from gbgo.utils import print_board, print_move
from gbgo import gotypes
from gbgo import scoring
from KGS_import import sgf_worker, inf_generate, consolidate_games

# To clear the board from the console
def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

# main
def main():
    t_0 = time.clock()
    random.seed(os.getpid() + int(time.clock()))
    np.random.seed(os.getpid() + int(time.clock()))
    timestr = dt.now().strftime("%Y%M%d_%H%M%S")

    # Game Parameters
    # eventually change to argparse
    board_size = 19
    sl_games = 10
    sl_epochs = 5
    sl_batch_size = 8

    # Checkpoints Storage
    chk_dir = "./agents/checkpoints"
    if not os.path.isdir(chk_dir):
         os.makedirs(chk_dir)

    # Initialize Encoder
    board_shape = (board_size, board_size)
    classes = board_size ** 2
    encoder = SevenPlaneEncoder(board_shape)
    input_shape = (encoder.num_planes, ) + board_shape
    data_dir = "./data"
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # SL File Storage
    sl_dir = "./agents/sl"
    if not os.path.isdir(sl_dir):
        os.makedirs(sl_dir)
    sl_file = "/gb_sl_0.h5"

    # Download and Encode KGS Game data
    # processor = GoDataProcessor(encoder=encoder.name(), data_directory=data_dir)
    # Data init
    # x, y = processor.load_go_data(num_samples=sl_samples)

    sgf_file = data_dir + '/all_sgfs.py'
    load = [line.strip() for line in open(sgf_file, 'r')]
    random.shuffle(load)
    train_data = [eval(t)[0].decode('ascii') for t in load[:sl_games]]
    test_data = [eval(t)[0].decode('ascii') for t in load[sl_games:sl_games*2]]

    print("Sampling Training List.")
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    train_results = [pool.apply_async(sgf_worker, args=(f, encoder, data_dir)) for f in train_data]
    train_games = [r.get() for r in train_results]
    pool.close()
    pool.join()

    print("Sampling Testing List.")
    pool = multiprocessing.Pool(processes=cores)
    test_results = [pool.apply_async(sgf_worker, args=(f, encoder, data_dir)) for f in test_data]
    test_games = [r.get() for r in test_results]
    pool.close()
    pool.join()

    predictor_agent = None
    train_file_base = data_dir + '/train_' + timestr
    test_file_base = data_dir + '/test_' + timestr

    # Generator Init
    train_samples = consolidate_games(train_games, board_size**2, train_file_base)
    test_samples = consolidate_games(test_games, board_size**2, test_file_base)

    train_generator = inf_generate(train_file_base, sl_batch_size, board_size**2)
    test_generator = inf_generate(test_file_base, sl_batch_size, board_size**2)

    if predictor_agent is None:
        # Configure Base Model
        predictor_model = Sequential()
        network_layers = large.layers(input_shape)
        for layer in network_layers:
            predictor_model.add(layer)

        # Configure model for predictor agent for supervised learning
        predictor_model.add(Flatten())
        predictor_model.add(Dense(1024))
        predictor_model.add(Activation('relu'))
        predictor_model.add(Dense(classes, activation="softmax"))
    else:
        predictor_model = predictor_agent.model

    # Use GPU with Tensorflow
    try:
        with tf.device("GPU:0"):
            opt = Adadelta()
            predictor_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # Data fit
            # predictor_model.fit(x,y, batch_size=sl_batch_size, epochs=sl_epochs, verbose=1)

            # Generator fit
            predictor_model.fit_generator(
                generator=train_generator,
                epochs=sl_epochs,
                steps_per_epoch=(train_samples / sl_batch_size),
                validation_data=test_generator,
                validation_steps=(test_samples / sl_batch_size),
                callbacks=[ModelCheckpoint(chk_dir + sl_file.replace(".h5", "_{epoch}.h5"))],
                verbose=1
            )
    except RuntimeError as e:
        print(str(e) + " prevented prediction model compile and fit.")

    # Create and Serialize DeepLearningAgent from predictor model
    predictor_agent = DeepLearningAgent(predictor_model, encoder)
    with h5py.File(sl_dir + sl_file, "w") as sl_out:
        predictor_agent.serialize(sl_out)

    # # RL File Storage
    # rl_dir = "./agents/rl"
    # if not os.path.isdir(rl_dir):
    #     os.makedirs(rl_dir)
    # rl_file_xp = "/gb_xp_0.h5"
    # rl_file_agent = "/gb_rl_0.h5"
    #
    # # Create another agent from predictor model
    # with h5py.File(sl_dir + sl_file, "r") as sl_in:
    #     sl_agent = load_prediction_agent(sl_in)
    #     sl_opponent = load_prediction_agent(sl_in)
    #
    # learning_agent = PolicyAgent(sl_agent.model, encoder)
    # opponent_agent = PolicyAgent(sl_opponent.model, encoder)

    # # Basic Experience Training
    # rl_num_games = 500
    # experience = experience_simulation(rl_num_games, learning_agent, opponent_agent)
    # with h5py.File(rl_dir + rl_file_xp, "w") as xp_out:
    #     experience.serialize(xp_out)
    #
    # try:
    #     with tf.device('GPU:0'):
    #         learning_agent.train(experience)
    # except RuntimeError as e:
    #     print(str(e) + " prevented learning agent training.")
    # with h5py.File(rl_dir + rl_file_agent, "w") as agent_out:
    #     learning_agent.serialize(agent_out)

    # Value files storage
    val_dir = "./agents/val"
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)
    val_file = "/gb_val_0.h5"

    # # Value Network starts with Base Model from above and follows the AlphaGo Network
    # value_network = Sequential()
    # network_layers = large.layers(input_shape)
    # for layer in network_layers:
    #     value_network.add(layer)
    # value_network.add(Conv2D(32, (3, 3), padding="same",
    #                          data_format='channels_first', activation='relu'))
    # value_network.add(Conv2D(filters=1, kernel_size=1, padding="same",
    #                          data_format='channels_first', activation='relu'))
    # value_network.add(Flatten())
    # value_network.add(Dense(256, activation='relu'))
    # value_network.add(Dense(1, activation='tanh'))

    # value_agent = ValueAgent(value_network, encoder)
    # try:
    #     with tf.device("GPU:0"):
    #         value_agent.train(experience)
    # except RuntimeError as e:
    #     print(str(e) + " prevented value agent training.")
    #
    # with h5py.File(val_dir + val_file, 'w') as val_agent_out:
    #     value_agent.serialize(val_agent_out)

    # Play game with against itself with GoBrainer Agent vs Strong-Policy
    # try:
    #     with tf.device("GPU:0"):
    #         fast_policy = load_prediction_agent(h5py.File(sl_dir + sl_file, "r"))
    #         p1_strong_policy = load_policy_agent(h5py.File(rl_dir + rl_file_agent, "r"))
    #         p1_value = load_value_agent(h5py.File(val_dir + val_file, "r"))
    #         gb_1 = GBAgent(p1_strong_policy.model, p1_value.model, encoder)
    #
    #         p2_strong_policy = load_policy_agent(h5py.File(rl_dir + rl_file_agent, "r"))
    #         p2_value = load_value_agent(h5py.File(val_dir + val_file, "r"))
    #
    #         game = GameState.new_game(board_size)
    #         while not game.is_over():
    #             clear()
    #             print_board(game.board)
    #             if game.next_player == gotypes.Player.black:
    #                 move = gb_1.select_move(game)
    #             else:
    #                 predicted_value = p2_value.predict(game)
    #                 move = p2_strong_policy.select_move(game)
    #             print_move(game.next_player, move)
    #             time.sleep(0.05)
    #             game = game.apply_move(move)
    #         game_result = scoring.compute_game_result(game)
    #         print(game_result)
    # except RuntimeError as e:
    #     print(e)

    hours, rem = divmod(time.clock*() - t_0, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))




if __name__ == '__main__':
    freeze_support()
    main()