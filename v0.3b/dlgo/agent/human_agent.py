# -*- coding: utf-8 -*-

import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point
from re import match
from dlgo import goboard
from dlgo.utils import point_from_coords


class HumanAgent(Agent):
    def select_move(self, game_state):
        move = goboard.Move.pass_turn()
        valid_move = False
        while not valid_move:
            try:
                human_move = input('-- ').upper()
                if match("P(ASS)*$", human_move):
                    move = goboard.Move.pass_turn()
                elif match("R(ESIGN)*$", human_move):
                    move = goboard.Move.resign()
                else:
                    point = point_from_coords(human_move.strip())
                    move = goboard.Move.play(point)

                valid_move = game_state.is_valid_move(move)
                if not valid_move:
                    print("Invalid move")

            except AssertionError:
                print("Invalid move")
            except ValueError:
                print("Invalid move")
            except IndexError:
                print("Invalid move")
        return move
