from scipy import stats

class Tester:
    def accept(self, p, y):
        for ix, iy in zip(p, y):
            self.value += float(self.accumulate(ix, iy))
        self.count += p.shape[0]

    def finish(self):
        ret = self.separate(self.value, self.count)
        self.value = 0
        self.count = 0
        return ret

    def accumulate(self, p, y):
        pass

    def separate(self, v, n):
        pass

    def __init__(self) -> None:
        self.value = 0
        self.count = 0


class RMSE(Tester):
    def accumulate(self, p, y):
        return ((p - y) ** 2)[0]

    def separate(self, v, n):
        return (v / n) ** 0.5


class MeanCorr(Tester):
    def accumulate(self, p, y):
        return stats.pearsonr(p, y)[0]

    def separate(self, v, n):
        return v / n
