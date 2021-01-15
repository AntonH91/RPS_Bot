from rps.agent.agents import RepeatAgent, HumanAgent
from rps.game.GameRunner import GameRunner
from rps_ml.baseline.rpscontest_agent import DLLU1Agent, MetaFixAgent

NUM_ROUNDS = 1000
NUM_GAMES = 100

p1 = MetaFixAgent()
p2 = DLLU1Agent()

game = GameRunner(p1, p2, num_rounds=NUM_ROUNDS)

score = {p1: 0, p2: 0}
for i in range(NUM_GAMES):
    print("\rPlaying game %d / %d" % (i+1, NUM_GAMES), end="")
    while not game.game_over:
        game.play_round()
    winner = game.winner
    if winner is not None:
        score[winner] += 1
    game.reset()
print()
print(score)
