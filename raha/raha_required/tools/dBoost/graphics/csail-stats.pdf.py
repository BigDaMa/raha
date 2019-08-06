#! /usr/bin/env python3

from matplotlib import pyplot
from utils import TANGO, rcparams, to_inches, filename
import numpy as np

batch, fname = filename("csail-stats.pdf")

real_outliers = {} # from the annotated set

bad_outliers     = {}
good_outliers    = {}
dubious_outliers = {}
ghost_outliers   = {}
total            = {}

bad     = 0
good    = 1
dubious = 2
ghost   = 3

dictionaries = [bad_outliers, good_outliers, dubious_outliers, ghost_outliers]

# Real annotations
with open('../datasets/real/csail-stats-annotated.txt', 'r') as f_true:
    for line in f_true:
        info = line.strip().split('\t')
        linum = int(info[0])
        cat   = int(info[len(info) - 1])
        real_outliers[linum] = cat

        if cat == 1:
            for x in info[1:len(info) - 1]:
                problem = x.split(':')[1]
                pos = ghost_outliers.get(problem,[])
                pos.append(linum)
                ghost_outliers[problem] = pos

# results
with open('../results/csail/csail-stats.txt', 'r') as f:
    for line in f:
        info  = line.strip().split('\t')
        linum = int(info[0])
        outlier_status = real_outliers.get(linum, 0)
        dic = dictionaries[outlier_status]

        for x in info[1:]:
            problem = x.split(':')[1]
            dic[problem] = dic.get(problem, 0) + 1
            total[problem] = total.get(problem, 0) + 1

            if outlier_status == 1:
                lst = ghost_outliers.get(problem, [])
                try: # the outlier was detected
                    pos = ghost_outliers.get(problem, [])
                    pos.remove(linum)
                    ghost_outliers[problem] = pos
                except ValueError:
                    pass

for key in ghost_outliers:
    ghost_outliers[key] = len(ghost_outliers[key])

#Plotting...
y_labels = [k for k in sorted(total, key = total.get, reverse = True)]
y_pos    = np.arange(len(y_labels))
data     = np.array([[dictionaries[i].get(k,0) for k in y_labels] for i in [good, dubious, bad, ghost]])

rcparams()

fig = pyplot.figure(figsize = (7.75, to_inches(100)))
ax = fig.add_subplot(111)

colors = [TANGO["green"][2], TANGO["grey"][1], TANGO["orange"][1], TANGO["red"][1]]
patch_handles = []
left = np.zeros(len(y_labels))
for i, d in enumerate(data):
    patch_handles.append(ax.barh(y_pos, d, color=colors[i%len(colors)], align='center', left=left, height=.6, linewidth = .5))
    left += d

for tic in ax.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False

ax.legend(['Outliers', 'Dubious', 'False positives', 'False negatives'], ncol = 2)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels)

if batch:
    pyplot.savefig(fname, transparent = True)

else:
    pyplot.show()
