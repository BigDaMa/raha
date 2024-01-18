import collections
import sys
import heapq
from ..utils import tupleops
from ..utils.printing import hhistplot

class Histogram:
    ID = "histogram"
    MAX_HIST_SIZE = 16

    def __init__(self, peak_threshold, outlier_threshold):
        self.peak_threshold = peak_threshold
        self.outlier_threshold = outlier_threshold

        self.all_counters = None
        self.counters = None
        self.sizes = None

    @staticmethod
    def register(parser):
        parser.add_argument("--" + Histogram.ID, nargs = 2,
                            metavar = ("peak_s", "outlier_s"),
                            help = "Use a discrete histogram-based model, identifying fields that" +
                            "have a peaked distribution (peakiness is determined using the peak_s " +
                            "parameter), and reporting values that fall in classes totaling less than "
                            "outlier_s of the corresponding histogram. Suggested values: 0.8 0.2.")

    @staticmethod
    def from_parse(params):
        return Histogram(*map(float, params))

    @staticmethod
    def add(counter, x):
        if counter is not None:
            counter[x] += 1
            if len(counter) > Histogram.MAX_HIST_SIZE:
                counter = None
        return counter

    @staticmethod
    def NbPeaks(distribution):
        return max(1, min(3, len(distribution) // 2)) # TODO

    @staticmethod
    def IsPeaked(distribution, peak_threshold):
        if distribution is None or len(distribution) > Histogram.MAX_HIST_SIZE: # TODO
            return False
        else:
            nb_peaks = Histogram.NbPeaks(distribution)
            total_weight = sum(distribution.values())
            peaks_weight = sum(heapq.nlargest(nb_peaks, distribution.values()))
            return peaks_weight > peak_threshold * total_weight

    def is_peaked(self, distribution):
        return Histogram.IsPeaked(distribution, self.peak_threshold)

    def fit(self, Xs, analyzer):
        for X in Xs:
            self.fit_one(X)
        self.finish_fit()

    def fit_one(self, X):
        # TODO: discard too full counters as we count?
        self.counters = tupleops.defaultif(self.counters, X, collections.Counter)
        self.sizes = tupleops.zeroif(self.sizes, X)
        self.counters = tupleops.merge(self.counters, X, tupleops.id, Histogram.add)
        self.sizes = tupleops.merge(self.sizes, X, tupleops.not_null, tupleops.plus)

    def finish_fit(self):
        self.all_counters = self.counters
        self.counters = tupleops.merge(self.counters, self.counters, self.is_peaked, tupleops.keep_if)

    def find_discrepancies_in_features(self, field_id, features, counters, sizes, discrepancies):
        for feature_id, (xi, mi, si) in enumerate(zip(features, counters, sizes)):
            if mi == None: # Histogram discarded (too large)
                continue
            if mi.get(xi, 0) < self.outlier_threshold * si: # si is the total size of the histogram
                discrepancies.append(((field_id, feature_id),))

    def find_discrepancies(self, X, _):
        discrepancies = []

        for field_id, (x, m, s) in enumerate(zip(X, self.counters, self.sizes)):
            if field_id > 0:
                self.find_discrepancies_in_features(field_id, x, m, s, discrepancies)

        # Only look at correlated columns if row already passes simpler tests
        if len(discrepancies) == 0:
            self.find_discrepancies_in_features(0, X[0], self.counters[0],
                                                self.sizes[0], discrepancies)

        return discrepancies

    def more_info(self, discrepancy, description, X, indent = "", pipe = sys.stdout):
        assert(len(discrepancy) == 1)
        field_id, feature_id = discrepancy[0]
        highlighted = X[field_id][feature_id]
        counter = self.all_counters[field_id][feature_id]
        pipe.write(indent + "• histogram for {}:\n".format(description))
        hhistplot(counter, highlighted, indent + "  ", pipe)
