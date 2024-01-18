from .gaussian import Simple
from .discrete import Histogram
from .mixture import Mixture
from .discretepart import PartitionedHistogram

ALL = lambda: (Simple, Histogram, Mixture, PartitionedHistogram)
