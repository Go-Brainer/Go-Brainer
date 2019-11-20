# -*- coding: utf-8 -*-

# Types based on "Deep Learning and the Game of Go" chapter 3

from enum import Enum
from collections import namedtuple


class Player(Enum):
    black = 1
    white = 2
    
    @property
    def other(self):
        return Player.black if self == Player.white else Player.white
    
    
class Point(namedtuple('Point', 'row col')):
    # Returns a list of all points adjacent to self
    def neighbors(self):
        return [
                Point(self.row - 1, self.col),
                Point(self.row + 1, self.col),
                Point(self.row, self.col - 1),
                Point(self.row, self.col + 1)
        ]
