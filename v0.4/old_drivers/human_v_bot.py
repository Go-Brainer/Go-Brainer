import dlgo.agent.naive as naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
from os import name, system
from re import match


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = naive.RandomBot()
    print_board(game.board)

    while not game.is_over():
        # Since making a play is necessary for changing board state but also changes
        # next player we must save the current player
        player_before = game.next_player
        move = None
        if game.next_player == gotypes.Player.black:
            valid_move = False
            while not valid_move:
                try:
                    human_move = input('-- ').upper()
                    if match("PA(S)*", human_move):
                        move = goboard.Move.pass_turn()
                    elif match("RE(SIGN)*", human_move):
                        move = goboard.Move.resign()
                    else:
                        point = point_from_coords(human_move.strip())
                        move = goboard.Move.play(point)

                    valid_move = game.is_valid_move(move)
                    if not valid_move:
                        print("Invalid move")

                except AssertionError:
                    print("Invalid move")
                except ValueError:
                    print("Invalid move")
                except IndexError:
                    print("Invalid move")
            # end of human input loop
        else:
            move = bot.select_move(game)
        clear()
        game = game.apply_move(move)
        print_board(game.board)
        print_move(player_before, move)


if __name__ == '__main__':
    main()
