#!/usr/bin/env python3
from utils import filename, save2pdf, setup
from utils.plots_helper import sensors 
import matplotlib
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

make,fname = filename("sensor-gaus-plots.pdf")
dfile = "../datasets/real/intel/sensors-1000-dirty.txt"

# e, s, y, x
args = [
    (1,1.5,0,1),
    (1,1.5,2,3)]

pdf = PdfPages(fname)
for (e,s,y,x) in args:
    title = "Outliers in Sensor Data\nGaussian, stddev=" + str(s)
    ofile = "../results/sensors_dirty_stat" + str(e) + "_gaussian" + str(s) + ".out"
    setup()
    sensors(title,x,y,dfile,ofile)
    save2pdf(pdf)
pdf.close()
