#!/usr/bin/env python3
from utils import TANGO, filename, rcparams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot

FILL  = [TANGO["red"][0], TANGO["green"][1]]
EDGES = [TANGO["red"][1], TANGO["green"][2]]

def make_hist(ys, labels):
    W = 0.8
    plot_w = len(ys) - 1 + W

    left = (5 - plot_w) / 2
    xs = [x + left for x in range(len(ys))]

    colors     = [FILL[y > 1]  for y in ys]
    edgecolors = [EDGES[y > 1] for y in ys]

    pyplot.close()

    rcparams(7)
    pyplot.rcParams['xtick.major.pad'] = '2'

    pyplot.figure(figsize = (.75, .4))
    ax = pyplot.axes(frameon = True)

    ax.set_xlim(0, 5)
    for side in ('top', 'left', 'right'):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_visible(False)
    pyplot.xticks([x + W for x in xs], labels, rotation=45, ha='right')
    ax.xaxis.set_ticklabels([str(l) for l in labels])
    ax.tick_params(top=False, bottom=False)

    pyplot.bar(xs, ys, width = W, color = colors, edgecolor = edgecolors, linewidth=0.5)

batch, fname = filename("table-histograms.pdf")

if batch:
    pdf = PdfPages(fname)

YS = [[1,3,3],[2,1.01,2,2],[1,6],[1,6]]

LABELS = [["1970","2014","2015"], ["Wed.","Thu.","Fri.","Sun."], ["8", "12"], [r"nnn", r"nn"]]

for ys, labels in zip(YS, LABELS):
    make_hist(ys, labels)

    if batch:
        pyplot.savefig(pdf, pad_inches=1, format = 'pdf')
    else:
        pyplot.show()

if batch:
    pdf.close()
