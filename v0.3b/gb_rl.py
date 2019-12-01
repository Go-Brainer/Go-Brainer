# tag::load_opponents[]
from dlgo.agent.pg import PolicyAgent, load_policy_agent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.rl.simulate import experience_simulation
from dlgo.goboard_fast import GameState, Player, Point
from dlgo import scoring

import random
import tensorflow as tf
import numpy as np
import os
import h5py
import time
from collections import namedtuple

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return load_policy_agent(h5file)


def play_games(agent1_fname, agent2_fname, num_games, board_size):
    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent2 = load_agent(agent2_fname)

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

    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)

    encoder = SevenPlaneEncoder((19, 19))
    fname = 'gb_rl_A.h5'
    fname_base = fname.replace('.h5', '_')

    # sl_agent = load_prediction_agent(h5py.File(fname, 'r'))
    #sl_opponent = load_prediction_agent(h5py.File(fname, 'r'))

    #rl_agent = PolicyAgent(sl_agent.model, encoder)
    #opponent = PolicyAgent(sl_opponent.model, encoder)

    rl_agent = load_agent(fname)
    opponent = load_agent(fname)

    # end::load_opponents[]

    # tag::run_simulation[]
    num_games = 500
    opponent_fname = fname

    generation = 0
    while generation < 5:

        for j in range(5):
            xp_file = fname_base + 'xp_gen' + str(generation) + "_batch" + str(j) + '.h5'
            experience = experience_simulation(num_games, rl_agent, opponent)
            with h5py.File(xp_file, 'w') as exp_out:
                experience.serialize(exp_out)
            rl_agent.train(experience)
            del experience

        outf = fname_base + str(generation) + '.h5'
        with h5py.File(outf, 'w') as rl_agent_out:
            rl_agent.serialize(rl_agent_out)

        wins, losses = play_games(outf, opponent_fname, 120, 19)
        if wins >= 70:
            opponent = load_agent(outf)
            generation += 1

if __name__ == '__main__':
    main()