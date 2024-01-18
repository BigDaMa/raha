#!/usr/bin/env python3
import sys
from math import exp
from matplotlib import pyplot
from utils import TANGO, filename, to_inches, rcparams

from os.path import join, dirname
sys.path.append(join(dirname(__file__), '../'))

from dboost.models.discrete import Histogram
from dboost.models.discretepart import PartitionedHistogram

import matplotlib
rcparams()
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amssymb,calc}\usepackage{pifont}\newcommand{\cmark}{\ding{51}}\newcommand{\xmark}{\ding{55}}"]

HISTOGRAMS = [
    [1, 100], # True-False
    [44, 45, 49, 100, 101, 102], # Samll increments
    [exp(-(x/3)**2) for x in range(-5, 1)], # Exp shape
    [1, 1, 2, 4, 80, 82, 84, 88] # Many large columns
]

def normalize(hist):
    mx = max(hist)
    return [x / mx for x in hist]

HISTOGRAMS = [normalize(hist) for hist in HISTOGRAMS]
HIST_LENGTH = max(len(hist) for hist in HISTOGRAMS)

def is_peaked(hist, model):
    hist = dict(enumerate(hist))
    if model is Histogram:
        return model.IsPeaked(hist, 0.8)
    else:
        return model.IsPeaked(hist, 5, 0.8)

BAR_WIDTH = 0.8
PLOTS_PER_LINE = 4
DISABLED_AXES = ("left", "right", "top")
fig, axes = pyplot.subplots(len(HISTOGRAMS) // PLOTS_PER_LINE, PLOTS_PER_LINE, sharex = True, sharey = True, squeeze = False, frameon = False, figsize = (to_inches(504), 1.75))

ORANGE, GREEN = (TANGO[color][2] for color in ("orange", "green"))
DASHES = [[], (2, 1)]

def ax(hid):
    row, column = hid // PLOTS_PER_LINE, hid % PLOTS_PER_LINE
    return axes[row][column]

def drawline(axis, xx, yy, clip, color, dashes):
    line = Line2D(xx, yy, linewidth = 1, color=color, dashes=DASHES[dashes])
    axis.add_line(line) # Add must come before clip
    line.set_clip_path(clip)

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def makehatches(axis, bar, in_di_peak, in_dd_peak):
    if not in_di_peak and not in_dd_peak:
        return

    x0, y0 = bar.get_x(), bar.get_y()
    w, h = bar.get_width(), bar.get_height()
    x1, y1 = x0 + w, y0 + h
    h_dist = 0.05

    colors, dashes = [], []
    if in_di_peak:
        colors.append(GREEN)
        dashes.append(True)
    if in_dd_peak:
        colors.append(ORANGE)
        dashes.append(False)

    for hid in range(-3, int(h / h_dist) + 3):
        (xx0, yy0), (xx1, yy1) = (x0, y0 + hid * h_dist - h_dist), (x1, y0 + hid * h_dist + h_dist)
        drawline(axis, (xx0, xx1), (yy0, yy1), bar, colors[hid % len(colors)], dashes[hid % len(dashes)])

    # Re-draw the bar's border to remove untidy overlaps
    axis.add_patch(Rectangle((x0, y0), w, h, fill=False,
                             linewidth=bar.get_linewidth(),
                             edgecolor=bar.get_edgecolor(),
                             zorder=10))

def arrow(axis, x1, y1, x2, y2, arrow_style, color="black", line_style='solid'):
    transform = axis.get_xaxis_transform()
    # transform is needed because Using data coordinates without an explicit
    # tranform doesn't allow drawing outside of the axes. annotation_clip=False
    # would work too

    axis.annotate('', xy=(x1, y1), xycoords=transform,
                  xytext=(x2, y2), textcoords=transform,
                  arrowprops={'arrowstyle': arrow_style,
                              'linestyle': line_style,
                              'color': color})

def markers(axis, x1, x2, y, color, dashes, msize = 0.02):
    lines = (Line2D((x1, x2), (y, y), color=color, dashes=DASHES[dashes], clip_on=False),
             Line2D((x1, x1), (y - msize, y + msize), color=color, clip_on=False),
             Line2D((x2, x2), (y - msize, y + msize), color=color, clip_on=False))
    for line in lines:
        axis.add_line(line)

PEAKINESS = [r'\xmark~no peak', r'\cmark~peak']
ALGOS = ["$D$-independent:", "$D$-dependent:", "Distribution-independent:", "Distribution-dependent:"]
FLABEL = r'\renewcommand{{\tabcolsep}}{{1pt}}\begin{{tabular}}{{ll}}{} & {}\\{} & {}\end{{tabular}}'

for hid, hist in enumerate(HISTOGRAMS):
    left = HIST_LENGTH / 2 - len(hist) / 2
    xs = [left + offset for offset in range(len(hist))]

    axis = ax(hid)
    bars = axis.bar(xs, hist, width=BAR_WIDTH)

    axis.xaxis.set_ticks([])
    axis.yaxis.set_ticks([])
    axis.yaxis.set_ticks_position('none')
    for side in DISABLED_AXES:
        axis.spines[side].set_color('none')
    axis.set_xlim(xmin=-1, xmax=HIST_LENGTH + 1)
    axis.set_ylim(ymin=0, ymax=1.05)
    axis.patch.set_visible(False)

    delta, min_hi, max_low, dd_first_hi = PartitionedHistogram.PeakProps(hist)
    di_nb_peaks = Histogram.NbPeaks(hist)
    di_first_hi = len(bars) - di_nb_peaks
    models_agree = dd_first_hi == di_first_hi

    annot_x = left + dd_first_hi - 1 + BAR_WIDTH / 2
    annot_y = (hist[dd_first_hi - 1] + hist[dd_first_hi]) / 2
    arrowstyle, dy = '<->', 0
    if hist[dd_first_hi] - hist[dd_first_hi - 1] < 0.2:
        arrowstyle += ',head_length=0.1'
        dy = 0.025
    arrow(axis, annot_x, hist[dd_first_hi - 1], annot_x, hist[dd_first_hi] + dy, arrowstyle)
    annot_color = 'black' # GREEN if delta >= 5 else RED '$\boldsymbol\times\mathbf{{{:.0f}}}$'
    axis.text(annot_x - 0.5, annot_y, r'$\times{:.0f}$'.format(delta), horizontalalignment='right',
              verticalalignment='center', color=annot_color, fontsize=14)
    for bid, bar in enumerate(bars):
        bar.set_color(TANGO['grey'][0])
        bar.set_edgecolor(TANGO['black'][2])
        bar.set_linewidth(1)
        makehatches(axis, bar, bid >= di_first_hi, bid >= dd_first_hi)

    peaked = lambda model: PEAKINESS[is_peaked(hist, model)]
    header = lambda dist_dep: ALGOS[dist_dep]
    label = FLABEL.format(header(False), peaked(Histogram),
                          header(True), peaked(PartitionedHistogram))
    axis.set_xlabel(label)

    di_x_start, dd_x_start = bars[di_first_hi].get_x(), bars[dd_first_hi].get_x()
    x_end = bars[-1].get_x() + BAR_WIDTH
    y_di, y_dd = 1.1, 1.2
    markers(axis, di_x_start, x_end, y_di, GREEN, True)
    markers(axis, dd_x_start, x_end, y_dd, ORANGE, False)

legend = fig.legend(handles = [Line2D((0, 50), (0, 0), dashes=DASHES[True], color=GREEN),
                               Line2D((0, 50), (0, 0), dashes=DASHES[False], color=ORANGE)],
                    labels = ["top bins ($D$-independent model)",
                              "top bins ($D$-dependent model)"],
                    loc = 8, ncol = 2, bbox_to_anchor = (0.5,-0.15), bbox_transform = fig.transFigure)

fig.tight_layout()
batch_mode, fname = filename("peakiness.pdf")

if batch_mode:
    fig.savefig(fname, transparent=True)
else:
    pyplot.show()
