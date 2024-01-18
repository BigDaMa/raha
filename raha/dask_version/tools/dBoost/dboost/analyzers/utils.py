from math import sqrt
from collections import namedtuple
from itertools import combinations, product

class Stats:
    MAX_CARDINALITY = 25 

    def __init__(self):
        self.sum = 0
        self.sum2 = 0
        self.min = float("+inf")
        self.max = float("-inf")
        self.count = 0
        self.elems = set()

    def update(self, x):
        if self == None:
            return

        self.sum += x
        self.sum2 += x * x
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        self.count += 1
        if self.elems != None:
            self.elems.add(x)
            if len(self.elems) > Stats.MAX_CARDINALITY:
                self.elems = None

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def sigma(self):
        return sqrt((self.sum2 / self.count) - (self.sum / self.count) ** 2)

    @property
    def cardinality(self):
        return len(self.elems) if self.elems != None else float("+inf")

    # TODO: This is not the most numerically stable formula
    @staticmethod
    def pearson(s1, s2, pw_prod):
        sigmas = s1.sigma * s2.sigma
        return ((pw_prod / s1.count - s1.avg * s2.avg) / sigmas
                if sigmas != 0 else None)

    def __repr__(self):
        FMT = "sum: {}, sum2: {}, min: {}, max: {}, count: {}, elems: {}, avg: {}, sigma: {}, card: {}"
        return FMT.format(self.sum, self.sum2, self.min, self.max, self.count, self.elems, self.avg, self.sigma, self.cardinality)
