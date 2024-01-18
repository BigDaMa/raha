import sys
from ..utils.tupleops import *
from ..utils.autoconv import autoconv

class Simple:
    ID = "gaussian"

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.model = None

    @staticmethod
    def register(parser):
        parser.add_argument("--" + Simple.ID, nargs = 1, metavar = "n_stdev",
                            help = "Use a gaussian model, reporting values that fall more than " +
                            "n_stdev standard deviations away from the mean. Suggested value: 3.")

    @staticmethod
    def from_parse(params):
        return Simple(*map(autoconv, params))

    def fit(self, Xs, analyzer):
        if analyzer.stats == None:
            print("Gaussian modelling requires a statistical preprocessing phase", file=sys.stderr)
            sys.exit(1)
        self.model = analyzer.stats

    def test_one(self, xi, stats):
        return stats == None or abs(xi - stats.avg) <= self.tolerance * stats.sigma

    def find_discrepancies(self, X, index):
        ret = []

        for field_id, (x, s) in enumerate(zip(X, self.model)):
            ret.extend(((field_id, test_id),) for (test_id, (xi, si))
                       in enumerate(zip(x, s)) if not self.test_one(xi, si))

        return ret

    INFO_FMT = "{feature_name}: {xi:.2g} falls out of range [{lo:.2f}, {hi:.2f}] = [{mu:.2f} - {t} * {sigma:.2f}, {mu:.2f} + {t} * {sigma:.2f}]\n"

    def more_info(self, discrepancy, description, X, indent = "", pipe = sys.stdout):
        assert(len(discrepancy) == 1)

        field_id, feature_id = discrepancy[0]
        feature_name = description[0]

        t = self.tolerance
        xi = X[field_id][feature_id]
        stats = self.model[field_id][feature_id]
        assert(stats != None)
        mu, sigma = stats.avg, stats.sigma
        lo, hi = mu - t * sigma, mu + t * sigma

        pipe.write(indent + Simple.INFO_FMT.format(**locals()))
