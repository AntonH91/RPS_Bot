from rps.constants import Move
from rps.agent.RPSAgent import RPSAgent


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