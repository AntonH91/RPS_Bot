from rps.agent.RPSAgent import RPSAgent
from rps.constants import Move, GameState
from rps.utils import utils


class GameRunner:

    def __init__(self, player_one: RPSAgent, player_two: RPSAgent, num_rounds=3):
        self.p1 = player_one
        self.p2 = player_two

        self.num_rounds = num_rounds

        self.played_rounds = 0
        self.game_log = None
        self.game_over = False

        self.winner = None

        self.reset()

    def reset(self):
        """Resets the GameRunner state to begin a new set of rounds."""
        self.played_rounds = 0
        self.game_log = {self.p1: GameLog(), self.p2: GameLog()}
        self.winner = None
        self.game_over = False
        self.p1.reset()
        self.p2.reset()

    def end_game(self):
        self.game_over = True
        if self.winner is None:
            self.winner = self.calc_winner()

    def calc_winner(self):
        p1_points = self.game_log[self.p1].points
        p2_points = self.game_log[self.p2].points

        if p1_points == p2_points:
            return None
        elif p1_points > p2_points:
            return self.p1
        else:
            return self.p2

    def play_round(self):
        """Plays a single round and returns the outcomes.

        @:return Tuple containing (p1_move, p2_move, round_winner, game_over)"""
        assert self.played_rounds < self.num_rounds and not self.game_over, \
            "Game has ended. Call reset() to begin a new game."

        # Fetch the last move
        if self.played_rounds > 0:
            p1_prev, p2_prev = [self.game_log[p].get_last_play()[0] for p in [self.p1, self.p2]]
        else:
            p1_prev = p2_prev = None

        # Compute the next moves
        p1_move = self.p1.play(p2_prev)
        p2_move = self.p2.play(p1_prev)

        # Report in the game states and do eventual disqualifications
        if self.report_state(self.p1, p1_move, p2_move) == GameState.DISQUALIFIED:
            self.winner = self.p2
            self.end_game()

        if self.report_state(self.p2, p2_move, p1_move) == GameState.DISQUALIFIED:
            self.winner = self.p1
            self.end_game()

        round_winner = utils.get_winner(self.p1, p1_move, self.p2, p2_move)

        self.played_rounds += 1

        if self.played_rounds == self.num_rounds:
            self.end_game()

        return p1_move, p2_move, round_winner, self.game_over

    def report_state(self, player: RPSAgent, player_move: Move, opposing_move: Move):
        """Computes the state after the moves have been played, reports to each player and writes to the log."""
        logger = self.game_log[player]

        game_state = utils.check_winner(player_move, opposing_move)

        logger.log_play(player_move, game_state)

        player.game_state(player_move, opposing_move, game_state)

        return game_state


class GameLog:
    def __init__(self):
        self.points = 0
        self.log = []

    def get_last_play(self):
        """Returns the move and outcome of the last played round.

        @:return Tuple (move: Move, outcome: GameState), or None if it is the first round"""

        if len(self.log) == 0:
            return None
        else:
            return self.log[-1]

    def log_play(self, move: Move, state: GameState):
        """Logs the move and outcome of the round"""
        self.log.append((move, state))
        if state == GameState.WIN:
            self.points += 1
