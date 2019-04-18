import csv
import matplotlib as mpl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from utils import TANGO

mpl.rcParams['text.latex.preamble'] = [r"\usepackage{siunitx}"]

LOF_SCALE, BASE_SIZE = 5, 10
MARKER_ALPHA, EDGE_WIDTH = 0.9, 0.1

def get_marker_params(color, **kwargs):
    return dict(facecolor="none", edgecolor=TANGO[color][2], alpha = MARKER_ALPHA, **kwargs)
    # must use facecolor: see https://github.com/matplotlib/matplotlib/issues/4083

INLIER3D = dict(linewidth=EDGE_WIDTH, s = BASE_SIZE, marker='o', **get_marker_params('green'))
OUTLIER3D = dict(linewidth=EDGE_WIDTH, s = BASE_SIZE * 2, marker='x', **get_marker_params('red'))
INLIER2D = dict(linewidth=EDGE_WIDTH, s = BASE_SIZE, **get_marker_params('green'))
OUTLIER2D = dict(linewidth=EDGE_WIDTH, **get_marker_params('red'))

sensors_schema = [r"Temperature (\si{\degreeCelsius})",r"Humidity (\si{\percent})",r"Light (\si{\lux})",r"Voltage (\si{\volt})"]
sensors_labels_padding= [2.5, 2.5, 3.25, 2.75]

def get_sensor_data(fname):
    d = [[],[],[],[]]
    with open(fname,'r') as f:
        for line in f:
            line = line.strip().split()
            for i in range(len(line)):
                d[i].append(float(line[i]))
    return d

def get_outliers_by_index(fname):
    d = []
    with open(fname,'r') as f:
        for line in f:
            line = line.strip()
            d.append(int(line))
    return d

def split_data(columns, outliers_indices):
    points = zip(*columns)
    outliers_indices = set(outliers_indices)

    inliers = [[] for _ in columns]
    outliers = [[] for _ in columns]

    for pid, point in enumerate(points):
        add_to = outliers if pid in outliers_indices else inliers
        for column, new_value in zip(add_to, point):
            column.append(new_value)

    return inliers, outliers

def scatter3D(ax, dataset, x, y, z, params, **kwargs):
    kwargs.update(params)
    sc = ax.scatter(dataset[x], dataset[y], dataset[z], **kwargs)
    sc.set_edgecolors = sc.set_facecolors = lambda *args: None

def sensors3D_one(fig, subplot_id, inliers, outliers, x, y, z):
    ax = fig.add_subplot(*subplot_id, projection='3d')

    align = ('right', 'left', 'left')
    axes = (ax.xaxis, ax.yaxis, ax.zaxis)
    xlabel, ylabel, zlabel = (sensors_schema[d] for d in (x, y, z))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # ax.elev = 15

    for (axis, align, colid) in zip(axes, align, (x, y, z)):
        # http://stackoverflow.com/questions/5525782/adjust-label-positioning-in-axes3d-of-matplotlib
        axis._axinfo['label']['space_factor'] = sensors_labels_padding[colid]
        axis._axinfo['ticklabel']['space_factor'] = 0.3
        for lbl in axis.get_ticklabels():
            lbl.set_horizontalalignment(align)

    NO_COLOR = (0, 0, 0, 0)
    for pane in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        pane.set_pane_color(NO_COLOR)

    scatter3D(ax, inliers, x, y, z, INLIER3D, s=BASE_SIZE / 2.0)
    scatter3D(ax, outliers, x, y, z, OUTLIER3D, s=BASE_SIZE / 2.0)

    NBINS = 7
    for ax_id in ('x', 'y'):
        ax.locator_params(axis=ax_id, nbins=NBINS)
    ax.autoscale(tight=True)

def sensors3D(dfile, ofile):
    from itertools import combinations
    from . import rcparams, to_inches
    rcparams()

    print(dfile, ofile)
    data = get_sensor_data(dfile)
    inliers, outliers = split_data(data, get_outliers_by_index(ofile))

    # mpl.rcParams["font.size"] = 9
    pyplot.close()

    columnsets = list(combinations(range(len(data)), 3))[:2]
    nb_plots = len(columnsets)

    COLUMNS = 1
    fig = pyplot.figure(figsize = (to_inches(200), to_inches(630 / 2.0))) #504|240, 666

    for subplot_id, columns in enumerate(sorted(columnsets)):
        sensors3D_one(fig, (nb_plots // COLUMNS, COLUMNS, 1 + subplot_id), inliers, outliers, *columns)

    pyplot.tight_layout(pad=2)

def sensors(title,x,y,dfile,ofile):
    d = get_sensor_data(dfile)
    outliers = get_outliers_by_index(ofile)
    ax = pyplot.gca()
    ax.set_title(title)
    ax.set_ylabel(sensors_schema[y])
    ax.set_xlabel(sensors_schema[x])
    o = [[],[],[],[]]
    for l in range(len(d[y])):
        if l in outliers:
            o[y].append(d[y][l])
            o[x].append(d[x][l])
    ax.scatter(d[x],d[y],**INLIER2D)
    ax.scatter(o[x],o[y], s=BASE_SIZE, **OUTLIER2D)

def lof(title,dfile,ofile):
    d = get_sensor_data(dfile)
    ax = pyplot.gca()
    ax.set_title(title)
    ax.set_xlabel(sensors_schema[1])
    ax.set_ylabel(sensors_schema[0])
    ax.scatter(d[1],d[0],**INLIER2D)
    with open(ofile,'r') as f:
        rdr = csv.reader(f,delimiter=' ')
        for l in rdr:
            l = [float(x) for x in l]
            if l[0] < 1.5: continue
            ax.scatter(l[2],l[1],s=(l[0]*LOF_SCALE),**OUTLIER2D)
