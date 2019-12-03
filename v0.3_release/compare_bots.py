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

from dlgo import agent
from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

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

def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size, gpu_frac = args

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

    with h5py.File(agent1_fname, 'r') as agent1f:
        agent1 = agent.load_prediction_agent(agent1f)
    with h5py.File(agent2_fname, 'r') as agent2f:
        agent2 = agent.load_prediction_agent(agent2f)

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
             num_games, num_workers, board_size):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac,
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
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins

def main():

    dir = './agents'

    agent1 = dir + '/deep_bot_B_0.h5'
    agent2 = dir + '/deep_bot_A0.h5'

    wins = evaluate(
        agent1, agent2,
        num_games=120,
        num_workers=4,
        board_size=19)
    print('Won %d / 120 games (%.3f)' % (
        wins, float(wins) / 120.0))

if __name__ == '__main__':
    main()