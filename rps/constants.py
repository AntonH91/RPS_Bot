from enum import Enum


class GameState(Enum):
    WIN = 1
    DRAW = 0
    LOSS = -1
    DISQUALIFIED = -2

class Move(Enum):
    ROCK = 1
    PAPER = 2
    SCISSORS = 3