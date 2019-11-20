import os
from gbgo import rl
from gbgo import scoring
from gbgo import goboard_fast as goboard
from gbgo.gotypes import Player

from collections import namedtuple
import h5py
import tempfile
import tensorflow as tf


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='tmp_xp_')
    os.close(fd)
    return fname

def simulate_game(black_player, white_player):
    moves = []
    game = goboard.GameState.new_game(19)
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


def experience_simulation(num_games, agent1, agent2):
    try:
        with tf.device('GPU:0'):
            collector1 = rl.ExperienceCollector()
            collector2 = rl.ExperienceCollector()
            xp_files = []
            xp = None
            color1 = Player.black
            mem_counter = 0
            for i in range(num_games):
                collector1.begin_episode()
                agent1.set_collector(collector1)
                collector2.begin_episode()
                agent2.set_collector(collector2)

                if color1 == Player.black:
                    black_player, white_player = agent1, agent2
                else:
                    white_player, black_player = agent2, agent1
                print("Game %d of %d" % (i+1, num_games))
                game_record = simulate_game(black_player, white_player)
                if game_record.winner == color1:
                    collector1.complete_episode(reward=1)
                    collector2.complete_episode(reward=-1)
                else:
                    collector2.complete_episode(reward=1)
                    collector1.complete_episode(reward=-1)
                color1 = color1.other
                mem_counter += 1
                if mem_counter < 10 or (i+1) < num_games:
                    if xp is None:
                        xp = rl.combine_experience([collector1, collector2])
                    else:
                        xp = rl.combine_experience([xp, collector1, collector2])
                else:
                    mem_counter = 0
                    tmp_file = get_temp_file()
                    xp_files.append(tmp_file)
                    with h5py.File(tmp_file, "w") as tmp_out:
                        xp.serialize(tmp_out)
                    xp = None
                collector1 = None
                collector2 = None
                collector1 = rl.ExperienceCollector()
                collector2 = rl.ExperienceCollector()

            first_file = xp_files[0]
            remaining_files = xp_files[1:]
            combined_buffer = rl.load_experience(h5py.File(first_file, "r"))
            for file in remaining_files:
                next_buffer = rl.load_experience(h5py.File(file, "r"))
                combined_buffer = rl.combine_experience([combined_buffer, next_buffer])

            for fname in xp_files:
                os.unlink(fname)
    except RuntimeError as e:
        print(e)

    return combined_buffer
