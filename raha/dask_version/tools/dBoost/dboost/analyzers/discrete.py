from collections import Counter
from itertools import combinations
from ..utils import tupleops
from ..utils.printing import debug
import sys

class DiscreteStats:
    ID = "discretestats"

    def __init__(self, max_buckets, fundep_size):
        assert(fundep_size >= 1)
        self.max_buckets = max_buckets
        self.fundep_size = fundep_size
        self.histograms = None
        self.stats = None
        self.hints = None

    @staticmethod
    def register(parser):
        parser.add_argument("--" + DiscreteStats.ID, nargs = 2, metavar = ("max_buckets", "fundep_size"),
                            help = "Find correlations using discrete histograms to count occurences of subtuples. Considers subtuples of size fundep_size, histograms are only retained if they total less than max_buckets distinct classes.")

    @staticmethod
    def from_parse(params):
        return DiscreteStats(*(int(param) for param in params))

    def fit(self, Xs):
        for (Xnum, X) in enumerate(Xs):
            if Xnum % 10 == 0 and sys.stdout.isatty():
                debug(Xnum, end='\r')

            if self.histograms == None:
                # types = tupleops.extract_types(X)
                self.histograms = {k: Counter() for k in tupleops.subtuple_ids(X, self.fundep_size)}

            # if not tupleops.types_consistent(types, X):
            #     print("line", Xnum, "has inconsistent types")
            #     for field_id in tupleops.compare_types(types, tupleops.extract_types(X)):
            #         print("Field", field_id, ":", X[field_id])
            #         print("...got :", tupleops.extract_types(X)[field_id])
            #         print("...exp :", types[field_id])

            to_remove = []
            for ids, hist in self.histograms.items():
                bucketkey = tuple(X[ix][isx] for (ix, isx) in ids)

                hist[bucketkey] += 1
                if len(hist) > self.max_buckets:
                    to_remove.append(ids)

            for ids in to_remove:
                del self.histograms[ids]

        print(len(self.histograms))
        self.hints = tuple(self.histograms.keys())

    def expand_stats(self):
        pass # This analyzer does not actually product stats
