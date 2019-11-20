from multiprocessing import freeze_support
import numpy as np
import os
import sys
import gzip
import tarfile
import shutil
import multiprocessing
import random
from datetime import datetime as dt
import time
from keras.utils import to_categorical
from glob import glob
from copy import deepcopy

from gbgo.data import KGSIndex
from gbgo.gosgf import Sgf_game
from gbgo.goboard_fast import Board, GameState, Move
from gbgo.gotypes import Player, Point
from gbgo.utils import print_board, print_move
from gbgo.scoring import compute_game_result
from gbgo.encoders import SevenPlaneEncoder

def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def extract_sgfs(data_dir, zip_file_name):
    pid_header = str(os.getpid()) + " "
    print(dt.now().strftime("%H:%M:%S.%f: ") + pid_header + "working on " + zip_file_name)
    this_gz = gzip.open(data_dir + '/' + zip_file_name)

    tar_file = zip_file_name[0:-3]
    this_tar = open(data_dir + '/' + tar_file, 'wb')
    shutil.copyfileobj(this_gz, this_tar)
    this_tar.close()

    zip_file = tarfile.open(data_dir + '/' + tar_file, 'r')
    name_list = zip_file.getnames()
    sgf_names = [name for name in name_list if name.endswith('.sgf')]
    game_counter = 0
    result = set()
    for sgf in sgf_names:
        # file_str = data_dir + '/' + tar_file[:-4] + str(counter) + '.sgf'
        file_str = data_dir + '/' + sgf[sgf.index('/')+1:] + str(game_counter)
        sgf_content = zip_file.extractfile(sgf).read()
        result.add((sgf_content,))

    print(dt.now().strftime("%H:%M:%S.%f: ") + pid_header + "completed " + zip_file_name)
    return result

def generate(file_base, batch_size, num_classes):
    base = file_base + '_features*.npz'
    for feature_file in glob(base):
        label_file = feature_file.replace('features', 'labels')
        x = np.load(feature_file)['arr_0']
        y = np.load(label_file)['arr_0']
        x = x.astype('float32')
        y = to_categorical(y.astype(int), num_classes)
        while x.shape[0] >= batch_size:
            x_batch, x = x[:batch_size], x[batch_size:]
            y_batch, y = y[:batch_size], y[batch_size:]
            yield x_batch, y_batch

def inf_generate(file_base, batch_size, num_classes):
    while True:
        for item in generate(file_base, batch_size, num_classes):
            yield item

def consolidate_games(game_list, num_classes, file_base):
    batch = 0
    total_moves = 0
    cur_moves = 0
    max_moves_per_batch = 90000
    features_list = []
    labels_list = []
    for game in game_list:
        x = np.load(game + "_features.npz")['arr_0']
        y = np.load(game + "_labels.npz")['arr_0']
        x = x.astype('float32')
        y = to_categorical(y.astype(int), num_classes)

        if cur_moves + x.shape[0] > max_moves_per_batch:
            features = np.concatenate(features_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            np.savez_compressed(file_base + "_features" + str(batch), features)
            np.savez_compressed(file_base + "_labels" + str(batch), labels)
            features_list = []
            labels_list = []
            cur_moves = 0
            batch += 1
        features_list.append(x)
        labels_list.append(y)
        cur_moves += x.shape[0]
        total_moves += x.shape[0]

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    np.savez_compressed(file_base + "_features" + str(batch), features)
    np.savez_compressed(file_base + "_labels" + str(batch), labels)
    return total_moves

def get_num_samples(game_list, num_classes):
    num_samples = 0
    for x,y in generate(game_list, num_classes):
        num_samples += x.shape[0]
    return num_samples

def process_sgf(sgf_string, encoder, data_dir):
    sgf = Sgf_game.from_string(sgf_string)
    date = sgf.get_date().decode('ascii')
    date = date[:date.find(',')]
    black_name = sgf.get_player_name('b')
    white_name = sgf.get_player_name('w')
    file_base = date + '_' + white_name + '-' + black_name
    if os.path.isfile(data_dir + '/' + file_base + '_features.npz'):
        return data_dir + '/' + file_base

    game_state, first_move_done = get_handicap(sgf)
    features = []
    labels = []

    # print_board(game_state.board)
    # clear()
    for item in sgf.main_sequence_iter():
        color, move_tuple = item.get_move()
        point = None
        if color is not None:
            player = Player.black if color == 'B' else Player.white

            if move_tuple is not None:
                row, col = move_tuple
                point = Point(row + 1, col + 1)
                move = Move.play(point)
            else:
                move = Move.pass_turn()
            game_state = game_state.apply_move(move)
            if first_move_done and point is not None:
                features.append(encoder.encode(game_state))
                labels.append(encoder.encode_point(point))
            # print_board(game_state.board)
            # print_move(player, move)
            # clear()
            first_move_done = True

    np.savez_compressed(data_dir + '/' + file_base + "_features", features)
    np.savez_compressed(data_dir + '/' + file_base + "_labels", labels)

    return data_dir + '/' + file_base

def sgf_worker(sgf_string, encoder, data_dir):
    try:
        return process_sgf(sgf_string, encoder, data_dir)
    except (KeyboardInterrupt, SystemExit):
        print('Exiting child.')

def worker(job):
    try:
        data_dir, filename = job
        return extract_sgfs(data_dir, filename)
    except (KeyboardInterrupt, SystemExit):
        print('Exiting child.')

def get_handicap(sgf):  # Get handicap stones
    go_board = Board(19, 19)
    first_move_done = False
    move = None
    game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
        first_move_done = True
        game_state = GameState(go_board, Player.white, None, move)
    return game_state, first_move_done

def main():
    data_dir = "./data"
    np.random.seed(os.getpid() + int(time.clock()))
    # if not os.path.isdir(data_dir):
    #     os.makedirs(data_dir)
    #
    # index = KGSIndex(data_directory=data_dir)
    # # index.download_files()
    #
    # files = []
    # for file_info in index.file_info:
    #     files.append(file_info['filename'])
    #
    # files_to_work = []
    # for f in files:
    #     files_to_work.append((data_dir, f))
    #
    # sgf_strings = []
    sgf_file = data_dir + '/all_sgfs.py'
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    # results = [pool.apply_async(worker, args=(f,)) for f in files_to_work]
    # results_list = [r.get() for r in results]
    # pool.close()
    # pool.join()
    # sgf_set = set.union(*results_list)
    #
    # with open(sgf_file, 'w') as f:
    #     for s in sgf_set:
    #         write_str = str(s) + '\n'
    #         f.write(write_str)

    load = [line.strip() for line in open(sgf_file, 'r')]
    random.shuffle(load)
    num_samples = 10
    train_data = [eval(t)[0].decode('ascii') for t in load[:num_samples]]
    test_data = [eval(t)[0].decode('ascii') for t in load[num_samples:]]

    t_0 = time.clock()
    encoder = SevenPlaneEncoder((19,19))
    all_features = []
    all_labels = []
    results = [pool.apply_async(sgf_worker, args=(f, encoder, data_dir)) for f in train_data]
    results_list = [r.get() for r in results]
    pool.close()
    pool.join()

    get_num_samples(results_list, 19**2)


    print(time.clock() - t_0)

if __name__ == '__main__':
    freeze_support()
    main()