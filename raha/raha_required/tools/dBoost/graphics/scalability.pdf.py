#!/usr/bin/env python3
from utils import filename, save2pdf, setup, rcparams, to_inches 
from utils.plots_helper import sensors 
import matplotlib
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import itertools

matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{siunitx}"]

make,fname = filename("scalability.pdf")

INTEL_TOTAL = 2313153
# labels: vary train size + algo type
# x: vary test size
# y: runtime in s

trs = [1000,100000]#,2313153]
tes = [INTEL_TOTAL]
tes_h = 2000000
_trs = ["1K","100K"]#,2313153]
_tes = [5000000,1000000,15000000,20000000]
#_tes = [0.100,1.000,10.000,100.000,1000.000,2313.153]
#es = ["1_gaussian1.5","0.7_mixture1_0.075","0.7_mixture2_0.075"]
es = [
    [1,"gaussian",1.5],
    [0.7,"mixture1",0.1],
    [0.7,"mixture2",0.05],
    [0.7,"histogram"]
]
# build data
results = {} 
vals = {} 

for (tr,te,e) in itertools.product(trs,tes,es):
    if (e[1],tr) not in results:
        results[(e[1],tr)] = []
        vals[(e[1],tr)] = []
    if e[1] == "gaussian":
        ofile = "../results/sensors_{}_stat{}_{}{}.out".format(tr,*e)
    elif e[1] == "histogram":
        ofile = "../results/csail/csail-timings-{}-{}.txt".format(tr,tes_h)
    else:
        ofile = "../results/sensors_{}_stat{}_{}_{}.out".format(tr,*e)
    with open(ofile,'r') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == "Time":
                #print("{} {} {}: {}".format(tr,e[1],float(line[1]),float(line[2])))
                vals[(e[1],tr)].append(float(line[1]))
                results[(e[1],tr)].append(float(line[2]))
            if line[0] == "Runtime":
                #print("{} {} {}: {}".format(tr,te,e[1],float(line[1])))
                vals[(e[1],tr)].append(te)
                results[(e[1],tr)].append(float(line[1]))
                continue
#print(results)
pdf = PdfPages(fname)
setup()
rcparams()
pyplot.gcf().set_size_inches(to_inches(240), to_inches(240)) # full column size is 240pt

ax = pyplot.gca()
ax.set_title("Scalability")
ax.set_xlabel("Test set size")
ax.set_ylabel("Runtime (s)")
lines = ["-","--"]
linecycler = itertools.cycle(lines)
ax.set_color_cycle(['g','g','r','r','b','b','m','m'])
ax.set_xlim([0,2000000])
for (e,(tr,_tr)) in itertools.product(es,zip(trs,_trs)):
    #vals[(e[1],tr)] = [val/1000 for val in vals[(e[1],tr)]]
    ax.plot(vals[(e[1],tr)],results[(e[1],tr)],next(linecycler),label = "{}, {}".format(e[1].capitalize(),_tr))#,marker='x',markersize=2.0)

ax.set_xticklabels(['0','0.5M','1M','1.5M','2.0M'])
ax.legend(loc=2,handlelength=3,prop={'size':6})
save2pdf(pdf)
pdf.close()
