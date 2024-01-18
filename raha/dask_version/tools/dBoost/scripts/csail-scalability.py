#!/usr/bin/env python3
import random
import shlex
import subprocess
import math

def scale_dataset(lines, dataset_size):
    scale_factor = int(math.ceil(dataset_size / len(lines)))
    lines = lines * scale_factor
    lines = lines[:dataset_size]
    return lines


def make_scaled_dataset(inpath, outpath, dataset_size):
    print("Making dataset of size {}...".format(dataset_size), end='')

    with open(inpath, mode="rb") as infile:
        lines = infile.readlines()

    lines = scale_dataset(lines, dataset_size)
    random.shuffle(lines)

    with open(outpath, mode="wb") as outfile:
        outfile.writelines(lines)

    print(" done.")
    return outpath

def make_large_dataset(dataset_size):
    inpath = '../datasets/real/csail.txt'
    outpath = '/tmp/csail-{}'.format(dataset_size)
    return make_scaled_dataset(inpath, outpath, dataset_size)

def make_trainset(training_size):
    inpath = '../datasets/real/csail.txt'
    outpath = '/tmp/csail-training-{}'.format(training_size)
    return make_scaled_dataset(inpath, outpath, training_size)

def get_args(training_path, dataset_path, printout_delay):
    cmd = "../dboost/dboost-stdin.py --statistical 1 --histogram 0.8 0.2 --in-memory --pr {} --train-with".format(printout_delay)
    args = shlex.split(cmd)
    args.append(training_path)
    args.append(dataset_path)
    return args

def time(training_sizes, dataset_size, printout_delay): # --pr
    dataset_path = make_large_dataset(dataset_size)
    for training_size in training_sizes:
        training_path = make_trainset(training_size)
        results_path = "../results/csail/csail-timings-{}-{}.txt".format(training_size, dataset_size)
        with open(results_path, mode="w") as resfile:
            args = get_args(training_path, dataset_path, printout_delay)
            print("Running: dataset size {}, training size {}...".format(dataset_size, training_size))
            subprocess.Popen(args, bufsize=-1, stderr=resfile, stdout=subprocess.DEVNULL).communicate()

#10k 10k 100k
time([x * 1000 for x in [1, 10, 100]], 2*1000*1000, 100000)
