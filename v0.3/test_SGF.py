from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import GameState, Move
from dlgo.gotypes import Point, Player
from dlgo.utils import print_board, print_move

sgf_content = "(;GM[1]FF[4]SZ[9];B[ee];W[ef];B[ff]" + \
            ";W[df];B[fe];W[fc];B[ec];W[gd];B[fb])"
sgf_game = Sgf_game.from_string(sgf_content)
game_state = GameState.new_game(9)
for item in sgf_game.main_sequence_iter():
    color, move_tuple = item.get_move()
    if color is not None and move_tuple is not None:
        player = Player.white if color == 'w' else Player.black
        row, col = move_tuple
        point = Point(row + 1, col + 1)
        move = Move.play(point)
        print_move(player, move)
        game_state = game_state.apply_move(move)
        print_board(game_state.board)

