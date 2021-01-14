from rps_ml.baseline.RPSContestAgent import ContestAgent
import random


# noinspection PyTypeChecker
class DLLU1Agent(ContestAgent):
    def __init__(self):
        super().__init__()
        self.numMeta = 6
        self.numPre = 30
        self.mScore = [5, 2, 5, 2, 4, 2]
        self.m = [random.choice("RPS")] * self.numMeta
        self.p = [random.choice("RPS")] * self.numPre
        self.length = 0
        self.best = [0, 0, 0]
        self.a = "RPS"
        self.rps = [1, 1, 1]
        self.soma = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.centripete = {'R': 0, 'P': 1, 'S': 2}
        self.centrifuge = {'RP': 0, 'PS': 1, 'SR': 2, 'PR': 3, 'SP': 4, 'RS': 5, 'RR': 6, 'PP': 7, 'SS': 8}
        self.pScore = [[5] * self.numPre, [5] * self.numPre, [5] * self.numPre, [5] * self.numPre, [5] * self.numPre,
                       [5] * self.numPre]
        self.moves = ['', '', '', '']
        self.beat = {'R': 'P', 'P': 'S', 'S': 'R'}
        self.limit = 8

    def reset(self):
        super().reset()
        self.numMeta = 6
        self.numPre = 30
        self.mScore = [5, 2, 5, 2, 4, 2]
        self.m = [random.choice("RPS")] * self.numMeta
        self.p = [random.choice("RPS")] * self.numPre
        self.length = 0
        self.best = [0, 0, 0]
        self.a = "RPS"
        self.rps = [1, 1, 1]
        self.soma = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.centripete = {'R': 0, 'P': 1, 'S': 2}
        self.centrifuge = {'RP': 0, 'PS': 1, 'SR': 2, 'PR': 3, 'SP': 4, 'RS': 5, 'RR': 6, 'PP': 7, 'SS': 8}
        self.pScore = [[5] * self.numPre, [5] * self.numPre, [5] * self.numPre, [5] * self.numPre, [5] * self.numPre,
                       [5] * self.numPre]
        self.moves = ['', '', '', '']
        self.beat = {'R': 'P', 'P': 'S', 'S': 'R'}
        self.limit = 8

    def contest_play(self):

        # see also www.dllu.net/self.rps
        # remember, rpsrunner.py is extremely useful for offline testing, 
        # here's self.a screenshot: http://i.imgur.com/DcO9M.png
        import random

        if not self.input:
            pass
        else:
            for i in range(self.numPre):
                pp = self.p[i]
                bpp = self.beat[pp]
                bbpp = self.beat[bpp]
                self.pScore[0][i] = 0.9 * self.pScore[0][i] + ((self.input == pp) - (self.input == bbpp)) * 3
                self.pScore[1][i] = 0.9 * self.pScore[1][i] + ((self.output == pp) - (self.output == bbpp)) * 3
                self.pScore[2][i] = 0.87 * self.pScore[2][i] + (self.input == pp) * 3.3 - (self.input == bpp) * 1.2 - (
                        self.input == bbpp) * 2.3
                self.pScore[3][i] = 0.87 * self.pScore[3][i] + (self.output == pp) * 3.3 - (
                        self.output == bpp) * 1.2 - (self.output == bbpp) * 2.3
                self.pScore[4][i] = (self.pScore[4][i] + (self.input == pp) * 3) * (1 - (self.input == bbpp))
                self.pScore[5][i] = (self.pScore[5][i] + (self.output == pp) * 3) * (1 - (self.output == bbpp))
            for i in range(self.numMeta):
                self.mScore[i] = 0.96 * (self.mScore[i] + (self.input == self.m[i]) - (
                        self.input == self.beat[self.beat[self.m[i]]]))
            self.soma[self.centrifuge[self.input + self.output]] += 1
            self.rps[self.centripete[self.input]] += 1
            self.moves[0] += str(self.centrifuge[self.input + self.output])
            self.moves[1] += self.input
            self.moves[2] += self.output
            self.length += 1
            for y in range(3):
                j = min([self.length, self.limit])
                while j >= 1 and not self.moves[y][self.length - j:self.length] in self.moves[y][0:self.length - 1]:
                    j -= 1
                i = self.moves[y].rfind(self.moves[y][self.length - j:self.length], 0, self.length - 1)
                self.p[0 + 2 * y] = self.moves[1][j + i]
                self.p[1 + 2 * y] = self.beat[self.moves[2][j + i]]
            j = min([self.length, self.limit])
            while j >= 2 and not self.moves[0][self.length - j:self.length - 1] in self.moves[0][0:self.length - 2]:
                j -= 1
            i = self.moves[0].rfind(self.moves[0][self.length - j:self.length - 1], 0, self.length - 2)
            if j + i >= self.length:
                self.p[6] = self.p[7] = random.choice("RPS")
            else:
                self.p[6] = self.moves[1][j + i]
                self.p[7] = self.beat[self.moves[2][j + i]]

            self.best[0] = self.soma[self.centrifuge[self.output + 'R']] * self.rps[0] / self.rps[
                self.centripete[self.output]]
            self.best[1] = self.soma[self.centrifuge[self.output + 'P']] * self.rps[1] / self.rps[
                self.centripete[self.output]]
            self.best[2] = self.soma[self.centrifuge[self.output + 'S']] * self.rps[2] / self.rps[
                self.centripete[self.output]]
            self.p[8] = self.p[9] = self.a[self.best.index(max(self.best))]

            for i in range(10, self.numPre):
                self.p[i] = self.beat[self.beat[self.p[i - 10]]]

            for i in range(0, self.numMeta, 2):
                self.m[i] = self.p[self.pScore[i].index(max(self.pScore[i]))]
                self.m[i + 1] = self.beat[self.p[self.pScore[i + 1].index(max(self.pScore[i + 1]))]]
        self.output = self.beat[self.m[self.mScore.index(max(self.mScore))]]
        if max(self.mScore) < 0.07 or random.randint(3, 40) > self.length:
            self.output = self.beat[random.choice("RPS")]
