#!/usr/bin/python
import os
import itertools

experiments = [
    [1,"gaussian",1.5],
    [0,"mixture",1,0.005],
    [0.7,"mixture",1,0.1],
    [0.7,"mixture",1,0.075],
    [0.7,"mixture",2,0.05],
    [0.7,"mixture",2,0.075]
]

os.chdir("../dboost")
for e in experiments:
    if e[1] == "gaussian":
        f = "sensors_dirty_stat{}_{}{}.out".format(*e)
        cmd = "python dboost-stdin.py --minimal -F ' ' ../datasets/real/intel/sensors-1000-dirty.txt --statistical {} --{} {} -d fracpart -d unix2date_float > ../results/{}".format(*(e+[f]))
    elif e[1] == "mixture":
        f = "sensors_dirty_stat{}_{}{}_{}.out".format(*e)
        cmd = "python dboost-stdin.py --minimal -F ' ' ../datasets/real/intel/sensors-1000-dirty.txt --statistical {} --{} {} {} -d fracpart -d unix2date_float > ../results/{}".format(*(e+[f]))
    else: assert(False)
    print(cmd)
    os.system(cmd)
    
