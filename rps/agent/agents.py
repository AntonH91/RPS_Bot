import random

from rps.constants import Move
from rps.agent.base_agent import RPSAgent


class HumanAgent(RPSAgent):
    moves = {
        "R": Move.ROCK,
        "P": Move.PAPER,
        "S": Move.SCISSORS
    }

    def play(self, opponent_move: Move) -> Move:
        return self.get_move()

    @classmethod
    def get_move(cls) -> Move:
        out_move = None
        while out_move is None:
            cmd = input("Enter move ([R]ock, [P]aper, [S]cissors): ").upper()
            if cmd in cls.moves:
                out_move = cls.moves[cmd]
            else:
                print("Bad move. Try again.")

        return out_move


class RepeatAgent(RPSAgent):

    def __init__(self):
        self.moves = {Move.ROCK: Move.PAPER,
                      Move.PAPER: Move.SCISSORS,
                      Move.SCISSORS: Move.ROCK}

        random.seed()

    def play(self, opponent_move) -> Move:

        if opponent_move is None:
            return random.choice([Move.ROCK, Move.PAPER, Move.SCISSORS])
        else:
            return self.moves[opponent_move]
