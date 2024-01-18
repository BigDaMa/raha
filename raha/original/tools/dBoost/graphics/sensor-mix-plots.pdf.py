#!/usr/bin/env python3
from utils import filename, save2pdf, setup
from utils.plots_helper import sensors 
from matplotlib.backends.backend_pdf import PdfPages

make, fname = filename("sensor-mix-plots.pdf")
dfile = "../datasets/real/intel/sensors-1000-dirty.txt"

# e, p, t, y, x
args = [
    (0.7,2,0.05,0,1),
    (0.7,2,0.05,0,3)]

pdf = PdfPages(fname)
for (e,p,t,y,x) in args:
    title = "Outliers in Sensor Data\n"+str(p)+" Gaussians,$\\theta$=" + str(t)
    ofile = "../results/sensors_dirty_stat" + str(e) + "_mixture" + str(p) + "_" + str(t) + ".out"
    setup()
    sensors(title,x,y,dfile,ofile)
    save2pdf(pdf)
pdf.close()
