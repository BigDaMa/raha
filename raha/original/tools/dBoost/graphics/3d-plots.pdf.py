#!/usr/bin/env python3
from utils import filename
from utils.plots_helper import sensors3D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot

batch, fname = filename("3d-plots.pdf")
dfile = "../datasets/real/intel/sensors-1000-dirty.txt"

MIX_OUTFILE = "../results/sensors_dirty_stat{}_mixture{}_{}.out"
GAUSS_OUTFILE = "../results/sensors_dirty_stat{}_gaussian{}.out"
LOF_OUTFILE = "../results/sensors_dirty_lof{}.out"


sources = [(GAUSS_OUTFILE, 1, 1.5),     # from gauss-plots
           # (MIX_OUTFILE, 0, 1, 0.005),  # from sensor-plots (p3)
           (MIX_OUTFILE, 0.7, 1, 0.1),  # from sensor-plots (p1 and p2)
           (MIX_OUTFILE, 0.7, 2, 0.05), # from sensor-mix-plots (p1 and p2)
           (LOF_OUTFILE, 2)] # from lof-plots

if batch:
    pdf = PdfPages(fname)

for out_template, *args in sources:
    ofile = out_template.format(*args)
    sensors3D(dfile,ofile)

    if batch:
        pyplot.savefig(pdf, pad_inches=1, format = 'pdf')
    else:
        pyplot.show()

if batch:
    pdf.close()
