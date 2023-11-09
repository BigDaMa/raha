import raha
import raha.dataset as dset
import raha.detection as dt
import time
import itertools
import os
import sys
import dask
from multiprocessing import shared_memory as sm
from distributed import Client, LocalCluster
import dask.dataframe as dd
from dask.distributed import get_client
import pickle
import pandas
import numpy
from dask.distributed import get_worker
from constants import *
import tempfile
import detection_parallel
import correction_parallel
import constants
import dataset_parallel as dp
import correction_single
import copy
import correction
import logging
import gc
from distributed.worker import Worker
from distributed.utils import get_ip


def run_raha(dataset_dictionary):
    print("________________")
    print("Running Raha...\n")

    #Run Raha and Benchmark
    raha = detection_parallel.DetectionParallel()
    detected_cells = raha.run_detection(dataset_dictionary)
    print("Detected {} cells!".format(len(detected_cells)))
    print("________________")
    return detected_cells


def run_baran(dataset_dictionary, detected_cells):
    print("________________")
    print("Running Baran...\n")
    baran = correction_parallel.CorrectionParallel()
    corrected_cells = baran.run(dataset_dictionary, detected_cells)
    print("Corrected {} cells!".format(len(corrected_cells)))
    print("________________")
    return corrected_cells


if __name__ == "__main__":

    dataset_dictionary = {
    "name": "flights",
    "path": "./../datasets/flights/dirty.csv",
    "clean_path": "./../datasets/flights/clean.csv",
    "results-folder": "./../datasets/flights/raha-baran-results-flights"
    }   
    dataset = dp.DatasetParallel(dataset_dictionary)
    detected_cells = run_raha(dataset_dictionary)
    corrected_cells = run_baran(dataset_dictionary, detected_cells)

    p, r, f = dataset.get_data_cleaning_evaluation(corrected_cells)[-3:]
    print("Total Performance on Data-Cleaning {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(dataset.name, p, r, f))

