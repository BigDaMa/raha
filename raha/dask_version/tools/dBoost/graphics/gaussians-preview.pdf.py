#!/usr/bin/env python3
from bisect import bisect_left
import matplotlib
from utils import TANGO, filename
from matplotlib import pyplot, mlab
from numpy import linspace

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['savefig.dpi'] = 300

def gaussian(x, mu, sigma):
    return mlab.normpdf(x, mu, sigma)

def gaussian_plt():
    pyplot.clf()

    x = linspace(-10, 10, num = 200)
    y = gaussian(x, 0, 1)

    NDEV = 1.5
    nlo = bisect_left(x, -NDEV)
    nhi = bisect_left(x, +NDEV)

    xlo, ylo = x[:nlo], y[:nlo]
    xmi, ymi = x[nlo-1:nhi+1], y[nlo-1:nhi+1]
    xhi, yhi = x[nhi:], y[nhi:]

    ax = pyplot.axes(frameon = False)
    ax.set_xlim((-5,5))

    pyplot.plot(xmi, ymi, color = TANGO["green"][2])
    pyplot.fill_between(xmi, 0, ymi, color = TANGO["green"][1])
    for (xx, yy) in ((xlo, ylo), (xhi, yhi)):
        pyplot.plot(xx, yy, color = TANGO["red"][1])
        pyplot.fill_between(xx, 0, yy, color = TANGO["red"][0])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    pyplot.tight_layout()

batch_mode, fname = filename("gaussians-preview.pdf")

if batch_mode:
    gaussian_plt()
    pyplot.savefig(fname, transparent = True)
