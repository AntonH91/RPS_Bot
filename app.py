from rps.agent.RepeatAgent import RepeatAgent
from rps.game.GameRunner import GameRunner
from rps_ml.baseline.RPS_Meta_Fix import MetaFixAgent
from rps_ml.baseline.DLLU1Agent import DLLU1Agent
p1 = MetaFixAgent()
p2 = DLLU1Agent()

game = GameRunner(p1, p2, num_rounds=1000)

score = {p1: 0, p2: 0}
for i in range(1000):
    print("\rPlaying game %d / %d" % (i+1, 1000), end="")
    while not game.game_over:
        game.play_round()
    winner = game.winner
    if winner is not None:
        score[winner] += 1
    game.reset()
print()
print(score)
