from tensorflow.python.keras.utils.np_utils import to_categorical

from rps.agent.base_agent import RPSAgent

import rps.constants as c

WINNING_MOVES = {
    c.Move.ROCK: c.Move.PAPER,
    c.Move.PAPER: c.Move.SCISSORS,
    c.Move.SCISSORS: c.Move.ROCK
}


def check_winner(played_move: c.Move, opposing_move: c.Move) -> c.GameState:
    """Compares two moves and returns a win state depending on the played move vs. the opposing move.

    @:param played_move Move enum describing the move by the active player.
    @:param opposing_move Move enum describing the move made by the opposing player
    @:return A GameState describing the round outcome for the player"""
    if played_move == opposing_move:
        return c.GameState.DRAW
    else:
        if opposing_move == WINNING_MOVES[played_move]:
            return c.GameState.LOSS
        else:
            return c.GameState.WIN


def get_winner(p1: RPSAgent, p1_move: c.Move, p2: RPSAgent, p2_move: c.Move):
    """Gets two pairs of agents and moves, then returns the winning agent.

    @:param p1 The agent representing Player 1
    @:param p1_move The move made by Player 1
    @:param p2 The agent representing Player 2
    @:param p2_move The move made by Player 2

    @:return The agent winning the round, or None if it is a draw.
    """
    state = check_winner(p1_move, p2_move)

    if state == c.GameState.WIN:
        return p1
    elif state in (c.GameState.LOSS, c.GameState.DISQUALIFIED):
        return p2
    else:
        return None


def get_categorical_move(move: c.Move):
    return to_categorical(move.value - 1, len(c.Move))
