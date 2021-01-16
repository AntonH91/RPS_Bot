from enum import Enum


class GameState(Enum):
    DRAW = 0
    WIN = 1
    LOSS = 2
    DISQUALIFIED = 3

class Move(Enum):
    ROCK = 1
    PAPER = 2
    SCISSORS = 3