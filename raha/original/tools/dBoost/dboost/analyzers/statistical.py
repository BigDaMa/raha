"""Statistical analyzer that collects various statistics for use by other
models.

It also detects correlations between columns using the Pearson R coefficient.
If the absolute value of the R value is greater than the epsilon
(corr_threshold) parameter, the two columns are reported as correlated.

Public members:

* stats: a tuple respecting the structure of the expanded input, with one Stats
         object per expanded field.

* pearsons: a dictionary mapping pairs of nested indices to correlation
            coefficients.

* hints: a list of correlated expanded fields ((x, subx), (y, suby))

"""

from numbers import Number
from math import fabs
from ..utils.tupleops import defaultif_masked, deepapply_masked, pair_ids, make_mask_abc
from ..analyzers.utils import Stats

class Pearson:
    ID = "statistical"

    def __init__(self, corr_threshold):
        self.corr_threshold = corr_threshold

        self.mask = None
        self.hints = []
        self.stats = None
        self.pearsons = {}
        self.pairwise_prods = None

    @staticmethod
    def register(parser):
        parser.add_argument("--" + Pearson.ID, nargs = 1, metavar = "epsilon",
                            help = "Use a statistical model analyzer, " +
                            "reporting correlated values with a pearson r" +
                            "value greater than epsilon.")

    @staticmethod
    def from_parse(params):
        return Pearson(*map(float, params))

    def pearson(self, pair_id):
        (idx, sidx), (idy, sidy) = pair_id
        return Stats.pearson(self.stats[idx][sidx], self.stats[idy][sidy],
                             self.pairwise_prods[pair_id])

    def fit(self, Xs):
        for X in Xs:
            if self.mask == None:
                self.mask = make_mask_abc(X, Number)

            self.stats = defaultif_masked(self.stats, X, Stats, self.mask)
            deepapply_masked(self.stats, X, Stats.update, self.mask)

            if self.pairwise_prods == None:
                self.pairwise_prods = {pid: 0 for pid in pair_ids(X, self.mask)}

            for (id1, id2) in self.pairwise_prods:
                (idx, sidx), (idy, sidy) = id1, id2
                self.pairwise_prods[(id1, id2)] += X[idx][sidx] * X[idy][sidy]

        for pair_id in self.pairwise_prods:
            pearson = self.pearson(pair_id)
            if pearson != None and fabs(pearson) > self.corr_threshold:
                self.hints.append(pair_id)
            self.pearsons[pair_id] = pearson

        self.hints.sort()

    def expand_stats(self):
        self.stats = ((None,) * len(self.hints),) + self.stats
