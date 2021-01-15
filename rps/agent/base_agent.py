from rps import constants


class RPSAgent:

    def reset(self):
        """Resets the agent to prepare for a new game."""
        pass

    def play(self, opponent_move: constants.Move) -> constants.Move:
        """Plays a single round, returns the move made by the agent, given the last move by the opponent.

        @:param opponent_move:int An integer representing the last move done by the opponent, as
        MOVE_ROCK, MOVE_SCISSORS or MOVE_PAPER.
        """

        return constants.Move.ROCK

    def game_state(self, own_move: constants.Move, opponent_move: constants.Move, result: constants.GameState):
        assert result in constants.GameState, \
            "Result must be GAME_WIN, GAME_DRAW or GAME_LOSS"
        RPSAgent.validate_move(own_move, "own_move must be a Move value")
        RPSAgent.validate_move(opponent_move, "opponent_move must be Move value")

    @classmethod
    def validate_move(cls, move, message):
        assert move in constants.Move, message

