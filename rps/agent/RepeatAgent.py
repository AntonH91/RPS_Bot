from rps.constants import Move
from rps.agent.RPSAgent import RPSAgent
import random


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
