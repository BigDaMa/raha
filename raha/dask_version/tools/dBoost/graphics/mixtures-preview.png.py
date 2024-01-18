#!/usr/bin/env python3
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['savefig.dpi'] = 300

def mixture_plt():
    from utils._multivariate import multivariate_normal

    centers = ([0.5, 1.5], [-1, -2], [2, -1])
    covs = ([[1.2, 2],[0.7, 1]], [[0.75, 0.6],[0.6, 0.75]], [[2, 0],[0, 2]])
    coeffs = (1,1,1)

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    RES = 500
    x = np.linspace(-5, 5, num = RES)
    y = np.linspace(-5, 5, num = RES)

    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    gaussians = (multivariate_normal(*param) for param in zip(centers, covs))
    z = sum(c * P for c, P in zip(coeffs, (g.pdf(pos) for g in gaussians)))

    from matplotlib import colors
    RED, GREEN = 0, 120
    HUES = (RED, GREEN)
    THRESHOLD = 0.1

    cmap_hsv = [(HUES[pos > THRESHOLD * 255] / 360,
                           0.25 + 0.75 * (pos / 255),
                           1 - 0.75 * (pos / 255))
                for pos in range(255)]
    cmap_rgb = colors.ListedColormap(colors.hsv_to_rgb(np.array([cmap_hsv]))[0])

    WHITE = (1, 1, 1, 0.05)
    STRIDE = 1
    ax.plot_surface(x, y, z, rstride = STRIDE, cstride = STRIDE, edgecolor = WHITE,
                    cmap = cmap_rgb, linewidth = 1, shade = True)
    ax.plot_wireframe(x, y, z, rstride = STRIDE, cstride = STRIDE, edgecolor = WHITE,
                      linewidth = 1)

    ax.dist = 7.5
    ax.elev = 20
    ax.set_axis_off()

from utils import filename
batch_mode, fname = filename("models-plots.png")

if batch_mode:
    mixture_plt()
    pyplot.savefig(fname, transparent = True)
