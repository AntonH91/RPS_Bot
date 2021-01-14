from rps.agent.HumanAgent import HumanAgent
from rps.agent.RepeatAgent import RepeatAgent

from rps.game.GameRunner import GameRunner

human = HumanAgent()
repeater = RepeatAgent()

game = GameRunner(human, repeater)

while not game.game_over:
    print(game.play_round())