TANGO = {"yellow": ("#fce94f", "#edd400", "#c4a000"),
         "orange": ("#fcaf3e", "#f57900", "#ce5c00"),
         "brown": ("#e9b96e", "#c17d11", "#8f5902"),
         "green": ("#8ae234", "#73d216", "#4e9a06"),
         "blue": ("#729fcf", "#3465a4", "#204a87"),
         "purple": ("#ad7fa8", "#75507b", "#5c3566"),
         "red": ("#ef2929", "#cc0000", "#a40000"),
         "grey": ("#eeeeec", "#d3d7cf", "#babdb6"),
         "black": ("#888a85", "#555753", "#2e3436")}

import sys
import matplotlib as mpl
from matplotlib import pyplot
from os.path import dirname, join

def filename(default):
    has_name = len(sys.argv) > 1
    return (has_name, sys.argv[1] if has_name else default)

def save2pdf(pdf):
    pyplot.tight_layout()
    pyplot.savefig(pdf, format = 'pdf')
    pyplot.clf()

def rcparams(fontsize = 9):
    mpl.rcParams.update({
        "font.size": fontsize,
        "font.family": "serif",
        "font.serif": "computer modern roman",
        "axes.titlesize": "medium",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.fontsize": "medium",
        "text.usetex": True,
        "text.latex.unicode": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })

def setup():
    rcparams()
    pyplot.gcf().set_size_inches(to_inches(200), to_inches(200)) # full column size is 240pt

def to_inches(points):
    return points / 72.26999
