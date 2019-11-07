########################################
# Benchmark
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# November 2019
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os

import numpy
import prettytable

import dataset
import detection
import baselines
import utilities
########################################


########################################
class Benchmark:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.RUN_COUNT = 10
        self.DATASETS = ["hospital", "flights", "beers", "movies_1", "tax"]

    def experiment_1(self):
        """
        This method conducts experiment 1.
        """
        print("------------------------------------------------------------------------\n"
              "-----------------Experiment 1: Comparison with Baselines----------------\n"
              "------------------------------------------------------------------------")
        results = {"Raha": {dn: {"Cell-Wise": [], "Tuple-Wise": []} for dn in self.DATASETS},
                   "dBoost": {dn: [] for dn in self.DATASETS},
                   "NADEEF": {dn: [] for dn in self.DATASETS},
                   "KATARA": {dn: [] for dn in self.DATASETS},
                   "ActiveClean": {dn: {"Cell-Wise": [], "Tuple-Wise": []} for dn in self.DATASETS}}
        detector = detection.Detection()
        detector.VERBOSE = False
        competitor = baselines.Baselines()
        for dataset_name in self.DATASETS:
            dataset_dictionary = {
                "name": dataset_name,
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
            }
            d = dataset.Dataset(dataset_dictionary)
            for r in range(self.RUN_COUNT):
                detection_dictionary = detector.run(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["Raha"][dataset_name]["Cell-Wise"].append(er)
                er = utilities.get_tuple_wise_evaluation(d, detection_dictionary)
                results["Raha"][dataset_name]["Tuple-Wise"].append(er)
                detection_dictionary = competitor.run_dboost(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["dBoost"][dataset_name].append(er)
                detection_dictionary = competitor.run_nadeef(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["NADEEF"][dataset_name].append(er)
                detection_dictionary = competitor.run_katara(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["KATARA"][dataset_name].append(er)
                detection_dictionary = competitor.run_activeclean(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["ActiveClean"][dataset_name]["Cell-Wise"].append(er)
                er = utilities.get_tuple_wise_evaluation(d, detection_dictionary)
                results["ActiveClean"][dataset_name]["Tuple-Wise"].append(er)
        t = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["dBoost"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["dBoost"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        row = ["NADEEF"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["NADEEF"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        row = ["KATARA"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["KATARA"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        row = ["ActiveClean"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["ActiveClean"][dataset_name]["Cell-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["Raha"][dataset_name]["Cell-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        print("Comparison with the stand-alone error detection tools. (Cell-wise precision, recall, f1 score)")
        print(t)
        t = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["ActiveClean"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["ActiveClean"][dataset_name]["Tuple-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["Raha"][dataset_name]["Tuple-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        t.add_row(row)
        print("Comparison in terms of detecting erroneous tuples. (Tuple-wise precision, recall, f1 score)")
        print(t)
        sampling_range = [20, 40, 60, 80, 100]
        results = {"Raha": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Min-k": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Maximum Entropy": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Metadata Driven": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS}}
        for dataset_name in self.DATASETS:
            dataset_dictionary = {
                "name": dataset_name,
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
            }
            d = dataset.Dataset(dataset_dictionary)
            for r in range(self.RUN_COUNT):
                for s in sampling_range:
                    detector.LABELING_BUDGET = s
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Raha"][dataset_name][s].append(er)
                    detection_dictionary = competitor.run_min_k(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Min-k"][dataset_name][s].append(er)
                    detection_dictionary = competitor.run_maximum_entropy(dataset_dictionary, s)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Maximum Entropy"][dataset_name][s].append(er)
                    detection_dictionary = competitor.run_metadata_driven(dataset_dictionary, s)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Metadata Driven"][dataset_name][s].append(er)
        t = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["Min-k"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Min-k"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        t.add_row(row)
        row = ["Maximum Entropy"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Maximum Entropy"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        t.add_row(row)
        row = ["Metadata Driven"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Metadata Driven"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        t.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Raha"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        t.add_row(row)
        print("Comparison with the error detection aggregators. (F1 score with the respective numbers of labeled tuples: {})".format(sampling_range))
        print(t)
########################################


########################################
if __name__ == "__main__":
    app = Benchmark()
    app.experiment_1()
########################################





