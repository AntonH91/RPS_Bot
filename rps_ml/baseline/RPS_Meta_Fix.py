from rps_ml.baseline.RPSContestAgent import ContestAgent
import random


class MetaFixAgent(ContestAgent):
    # Source: http://www.rpscontest.com/entry/5649874456412160
    # @author: TeleZ
    def __init__(self):
        super().__init__()

        self.skor2 = [0] * 6
        self.skor1 = [[0] * 18, [0] * 18, [0] * 18, [0] * 18, [0] * 18, [0] * 18]
        self.meta = [random.choice("RPS")] * 6
        self.prin = [random.choice("RPS")] * 18
        self.DNA = [""] * 3
        self.RNA = {'RR': '1', 'RP': '2', 'RS': '3', 'PR': '4', 'PP': '5', 'PS': '6', 'SR': '7', 'SP': '8',
                    'SS': '9'}
        self.mix = {'RR': 'R', 'RP': 'R', 'RS': 'S', 'PR': 'R', 'PP': 'P', 'PS': 'P', 'SR': 'S', 'SP': 'P', 'SS': 'S'}
        self.rot = {'R': 'P', 'P': 'S', 'S': 'R'}

    def reset(self):
        super().reset()
        self.skor2 = [0] * 6
        self.skor1 = [[0] * 18, [0] * 18, [0] * 18, [0] * 18, [0] * 18, [0] * 18]
        self.meta = [random.choice("RPS")] * 6
        self.prin = [random.choice("RPS")] * 18
        self.DNA = [""] * 3
        self.RNA = {'RR': '1', 'RP': '2', 'RS': '3', 'PR': '4', 'PP': '5', 'PS': '6', 'SR': '7', 'SP': '8',
                    'SS': '9'}
        self.mix = {'RR': 'R', 'RP': 'R', 'RS': 'S', 'PR': 'R', 'PP': 'P', 'PS': 'P', 'SR': 'S', 'SP': 'P', 'SS': 'S'}
        self.rot = {'R': 'P', 'P': 'S', 'S': 'R'}

    def contest_play(self):
        if not self.input:
            pass
        else:
            for j in range(18):
                for i in range(4):
                    self.skor1[i][j] *= 0.8
                for i in range(4, 6):
                    self.skor1[i][j] *= 0.5
                for i in range(0, 6, 2):
                    self.skor1[i][j] -= (self.input == self.rot[self.rot[self.prin[j]]])
                    self.skor1[i + 1][j] -= (self.output == self.rot[self.rot[self.prin[j]]])
                for i in range(2, 6, 2):
                    self.skor1[i][j] += (self.input == self.prin[j])
                    self.skor1[i + 1][j] += (self.output == self.prin[j])
                self.skor1[0][j] += 1.3 * (self.input == self.prin[j]) - 0.3 * (self.input == self.rot[self.prin[j]])
                self.skor1[1][j] += 1.3 * (self.output == self.prin[j]) - 0.3 * (self.output == self.rot[self.prin[j]])
            for i in range(6):
                self.skor2[i] = 0.9 * self.skor2[i] + (self.input == self.meta[i]) - (self.input == self.rot[self.rot[self.meta[i]]])
            self.DNA[0] += self.input
            self.DNA[1] += self.output
            self.DNA[2] += self.RNA[self.input + self.output]
            for i in range(3):
                j = min(21, len(self.DNA[2]))
                k = -1
                while j > 1 and k < 0:
                    j -= 1
                    k = self.DNA[i].rfind(self.DNA[i][-j:], 0, -1)
                self.prin[2 * i] = self.DNA[0][j + k]
                self.prin[2 * i + 1] = self.rot[self.DNA[1][j + k]]
                k = self.DNA[i].rfind(self.DNA[i][-j:], 0, j + k - 1)
                self.prin[2 * i] = self.mix[self.prin[2 * i] + self.DNA[0][j + k]]
                self.prin[2 * i + 1] = self.mix[self.prin[2 * i + 1] + self.rot[self.DNA[1][j + k]]]
            for i in range(6, 18):
                self.prin[i] = self.rot[self.prin[i - 6]]
            for i in range(0, 6, 2):
                self.meta[i] = self.prin[self.skor1[i].index(max(self.skor1[i]))]
                self.meta[i + 1] = self.rot[self.prin[self.skor1[i + 1].index(max(self.skor1[i + 1]))]]
        self.output = self.rot[self.meta[self.skor2.index(max(self.skor2))]]
