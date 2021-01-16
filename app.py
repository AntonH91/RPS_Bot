from rps.agent.agents import RepeatAgent, HumanAgent
from rps.game.GameRunner import GameRunner
from rps_ml.baseline.rpscontest_agent import DLLU1Agent, MetaFixAgent
from rps_ml.ai_agent.ai_agent import AiAgent
from rps_ml.ai_agent.models import DefaultGameplayModel, DefaultPredictionModel




NUM_ROUNDS = 1000
NUM_GAMES = 100

p1 = MetaFixAgent()
p2 = RepeatAgent()


gameplay_model = DefaultGameplayModel()
prediction_model = DefaultPredictionModel()

gameplay_model.compile(optimizer='rmsprop', loss='mse')
prediction_model.compile(optimizer=None, loss='categorical_crossentropy')

pai = AiAgent(prediction_model=prediction_model,
              gameplay_model=gameplay_model)


p1 = pai
p2 = MetaFixAgent()

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
