import sys
from collections import Counter, defaultdict
from ..utils.printing import hhistplot

class PartitionedHistogram:
    ID = "partitionedhistogram"

    def __init__(self, jmp_threshold, peak_threshold, outlier_threshold):
        self.jmp_threshold = jmp_threshold
        self.peak_threshold = peak_threshold
        self.outlier_threshold = outlier_threshold

        self.all_counters = None
        self.counters = None
        self.sizes = None

    @staticmethod
    def register(parser):
        parser.add_argument("--" + PartitionedHistogram.ID, nargs = 3,
                            metavar = ("jmp_threshold", "peak_s", "outlier_s"),
                            help = "TODO.")

    @staticmethod
    def from_parse(params):
        return PartitionedHistogram(*map(float, params))

    @staticmethod
    def add(counters, sizes,  x):
        key, val = x[0], x[1:]
        counters[key][val] += 1
        sizes[key] += 1
        # print("added", key, val, "to", counters, sizes)

    def fit(self, Xs, analyzer):
        for X in Xs:
            self.fit_one(X)
        self.finish_fit()

    def fit_one(self, X):
        correlations = X[0]
        if self.counters == None:
            self.counters = tuple(defaultdict(Counter) for _ in correlations)
            self.sizes = tuple(defaultdict(int) for _ in correlations)
        for c, s, cr in zip(self.counters, self.sizes, correlations):
            PartitionedHistogram.add(c, s, cr)
        # print(self.counters)

    @staticmethod
    def PeakProps(ys):
        delta, min_hi, max_low, start_hi = max((ys[i+1] / ys[i], ys[i+1], ys[i], i+1) for i in range(len(ys) - 1))
        return delta, min_hi, max_low, start_hi

    @staticmethod
    def IsPeaked(hist, jmp_threshold, peak_threshold):
        if len(hist) > 16 or len(hist) < 2:
            return False
        else:
            ys = sorted(hist.values())
            delta, _, _, start_hi = PartitionedHistogram.PeakProps(ys)
            sum_low, sum_hi = sum(ys[:start_hi]), sum(ys[start_hi:])
            # if len(ys) > 3:
            #     print(ys)
            #     print(delta, min_hi, max_low, start_hi, sum_low, sum_hi)
            return (delta > jmp_threshold and
                    sum_hi > peak_threshold * (sum_hi + sum_low))

    def finish_fit(self):
        self.all_counters = self.counters
        self.counters = tuple({k: vs for (k, vs)
                               in counters.items()
                               if PartitionedHistogram.IsPeaked(vs, self.jmp_threshold, self.peak_threshold)}
                              for counters in self.counters)
        # from pprint import pprint
        # pprint(self.all_counters)

    def find_discrepancies_in_features(self, field_id, features, discrepancies):
        for feature_id, (xi, mi, si) in enumerate(zip(features, self.counters, self.sizes)):
            k, v = xi[0], xi[1:]
            mi, si = mi.get(k, None), si[k]
            if mi != None and mi.get(v, 0) < self.outlier_threshold * si:
                discrepancies.append(((field_id, feature_id),))

    def find_discrepancies(self, X, _):
        discrepancies = []
        self.find_discrepancies_in_features(0, X[0], discrepancies)
        return discrepancies

    def more_info(self, discrepancy, description, X, indent = "", pipe = sys.stdout):
        assert(len(discrepancy) == 1)
        field_id, feature_id = discrepancy[0]
        assert(field_id == 0)
        xi = X[0][feature_id]
        key, val = xi[0], xi[1:]
        kdesc, vdesc = description[0], description[1:]
        histograms = self.all_counters[feature_id]
        pipe.write(indent + "• histogram for {} if '{}' = {}:\n".format(vdesc, kdesc, key))
        hhistplot(histograms[key], val, indent + "  ", pipe)

        if len(histograms) > 1:
            for k, hist in histograms.items():
                if k != key:
                    pipe.write(indent + "... if '{}' = {}:\n".format(kdesc, k))
                    hhistplot(hist, None, indent + "  ", pipe)
