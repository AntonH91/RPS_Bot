from rps.constants import Move
from rps.agent.RPSAgent import RPSAgent


class ContestAgent(RPSAgent):
    translations = {"R": Move.ROCK,
                    "P": Move.PAPER,
                    "S": Move.SCISSORS,
                    Move.ROCK: "R",
                    Move.PAPER: "P",
                    Move.SCISSORS: "S"}

    def __init__(self):
        self.input = None
        self.output = None

    def play(self, opponent_move: Move) -> Move:
        """Translates the move into RPSContest terms, plays a round and returns the calculated output."""
        if opponent_move is not None:
            self.input = self.translations[opponent_move]

        self.contest_play()

        return self.translations[self.output]

    def contest_play(self):
        pass
