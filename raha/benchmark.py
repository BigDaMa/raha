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
import sys
import time
import shutil

import numpy
import prettytable

import raha.dataset
import raha.detection
import raha.baselines
import raha.utilities
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
        self.DATASETS = ["hospital", "flights", "beers", "rayyan", "movies_1", "tax"]

    def experiment_1(self):
        """
        This method conducts experiment 1.
        """
        print("------------------------------------------------------------------------\n"
              "-----------------Experiment 1: Comparison with Baselines----------------\n"
              "------------------------------------------------------------------------")
        results = {"dBoost": {dn: [] for dn in self.DATASETS},
                   "NADEEF": {dn: [] for dn in self.DATASETS},
                   "KATARA": {dn: [] for dn in self.DATASETS},
                   "ActiveClean": {dn: {"Cell-Wise": [], "Tuple-Wise": []} for dn in self.DATASETS},
                   "Raha": {dn: {"Cell-Wise": [], "Tuple-Wise": []} for dn in self.DATASETS}}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        competitor = raha.baselines.Baselines()
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                detection_dictionary = detector.run(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                results["Raha"][dataset_name]["Cell-Wise"].append(er)
                er = raha.utilities.get_tuple_wise_evaluation(d, detection_dictionary)
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
                er = raha.utilities.get_tuple_wise_evaluation(d, detection_dictionary)
                results["ActiveClean"][dataset_name]["Tuple-Wise"].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["dBoost"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["dBoost"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_1.add_row(row)
        row = ["NADEEF"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["NADEEF"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_1.add_row(row)
        row = ["KATARA"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["KATARA"][dataset_name]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_1.add_row(row)
        row = ["ActiveClean"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["ActiveClean"][dataset_name]["Cell-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_1.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["Raha"][dataset_name]["Cell-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_1.add_row(row)
        table_2 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["ActiveClean"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["ActiveClean"][dataset_name]["Tuple-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_2.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            p, r, f = numpy.mean(numpy.array(results["Raha"][dataset_name]["Tuple-Wise"]), axis=0)
            row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
        table_2.add_row(row)
        sampling_range = [20, 40, 60, 80, 100]
        results = {"Min-k": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Maximum Entropy": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Metadata Driven": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Raha": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS}}
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for s in sampling_range:
                    detection_dictionary = competitor.run_min_k(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Min-k"][dataset_name][s].append(er)
                    detection_dictionary = competitor.run_maximum_entropy(dataset_dictionary, s)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Maximum Entropy"][dataset_name][s].append(er)
                    detection_dictionary = competitor.run_metadata_driven(dataset_dictionary, s)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Metadata Driven"][dataset_name][s].append(er)
                    detector.LABELING_BUDGET = s
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Raha"][dataset_name][s].append(er)
        table_3 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        row = ["Min-k"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Min-k"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        table_3.add_row(row)
        row = ["Maximum Entropy"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Maximum Entropy"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        table_3.add_row(row)
        row = ["Metadata Driven"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Metadata Driven"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        table_3.add_row(row)
        row = ["Raha"]
        for dataset_name in self.DATASETS:
            f_list = [numpy.mean(numpy.array(results["Raha"][dataset_name][s]), axis=0)[2] for s in sampling_range]
            row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
        table_3.add_row(row)
        print("Comparison with the stand-alone error detection tools. (Precision, recall, f1 score)")
        print(table_1)
        print("Comparison in terms of detecting erroneous tuples. (Tuple-wise precision, recall, f1 score)")
        print(table_2)
        print("Comparison with the error detection aggregators. (F1 score with the respective numbers of labeled tuples: {})".format(sampling_range))
        print(table_3)

    def experiment_2(self):
        """
        This method conducts experiment 2.
        """
        print("------------------------------------------------------------------------\n"
              "------------------Experiment 2: Feature Impact Analysis-----------------\n"
              "------------------------------------------------------------------------")
        results = {"TF-IDF": {dn: [] for dn in self.DATASETS},
                   "All - OD": {dn: [] for dn in self.DATASETS},
                   "All - PVD": {dn: [] for dn in self.DATASETS},
                   "All - RVD": {dn: [] for dn in self.DATASETS},
                   "All - KBVD": {dn: [] for dn in self.DATASETS},
                   "All": {dn: [] for dn in self.DATASETS}}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for feature_specification in results:
                    if feature_specification == "TF-IDF":
                        detector.ERROR_DETECTION_ALGORITHMS = ["TFIDF"]
                    if feature_specification == "All - OD":
                        detector.ERROR_DETECTION_ALGORITHMS = ["PVD", "RVD", "KBVD"]
                    if feature_specification == "All - PVD":
                        detector.ERROR_DETECTION_ALGORITHMS = ["OD", "RVD", "KBVD"]
                    if feature_specification == "All - RVD":
                        detector.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "KBVD"]
                    if feature_specification == "All - KBVD":
                        detector.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD"]
                    if feature_specification == "All":
                        detector.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results[feature_specification][dataset_name].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for feature_specification in results:
            row = [feature_specification]
            for dataset_name in self.DATASETS:
                p, r, f = numpy.mean(numpy.array(results[feature_specification][dataset_name]), axis=0)
                row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
            table_1.add_row(row)
        print("System effectiveness with different feature groups. (Precision, recall, f1 score)")
        print(table_1)

    def experiment_3(self):
        """
        This method conducts experiment 3.
        """
        print("------------------------------------------------------------------------\n"
              "-----------------Experiment 3: Sampling Impact Analysis-----------------\n"
              "------------------------------------------------------------------------")
        sampling_range = [5, 10, 15, 20, 25, 30]
        results = {"Uniform": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS},
                   "Clustering-Based": {dn: {s: [] for s in sampling_range} for dn in self.DATASETS}}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for s in sampling_range:
                    detector.LABELING_BUDGET = s
                    detector.CLUSTERING_BASED_SAMPLING = False
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Uniform"][dataset_name][s].append(er)
                    detector.CLUSTERING_BASED_SAMPLING = True
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Clustering-Based"][dataset_name][s].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for sampling_approach in results:
            row = [sampling_approach]
            for dataset_name in self.DATASETS:
                f_list = [numpy.mean(numpy.array(results[sampling_approach][dataset_name][s]), axis=0)[2] for s in sampling_range]
                row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
            table_1.add_row(row)
        print("System effectiveness with different sampling approaches. (F1 score with the respective numbers of labeled tuples: {})".format(sampling_range))
        print(table_1)

    def experiment_4(self):
        """
        This method conducts experiment 4.
        """
        print("------------------------------------------------------------------------\n"
              "------------Experiment 4: Strategy Filtering Impact Analysis------------\n"
              "------------------------------------------------------------------------")
        results = {"Without Strategy Filtering": {dn: [] for dn in self.DATASETS},
                   "With Strategy Filtering": {dn: [] for dn in self.DATASETS}}
        for dataset_name in self.DATASETS:
            dataset_dictionary = {
                "name": dataset_name,
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
            }
            d = raha.dataset.Dataset(dataset_dictionary)
            d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
            raha.utilities.dataset_profiler(d)
            raha.utilities.evaluation_profiler(d)
        # for r in range(self.RUN_COUNT):
        #     for dataset_name in self.DATASETS:
        #         dataset_dictionary = {
        #             "name": dataset_name,
        #             "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        #             "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
        #         }
        #         d = raha.dataset.Dataset(dataset_dictionary)
        #         selected_strategies = raha.utilities.get_selected_strategies(d, self.DATASETS)


    def experiment_5(self):
        """
        This method conducts experiment 5.
        """
        print("------------------------------------------------------------------------\n"
              "------------Experiment 5: User Labeling Error Impact Analysis-----------\n"
              "------------------------------------------------------------------------")
        user_labeling_error_range = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        results = {"Homogeneity Resolution": {dn: {e: [] for e in user_labeling_error_range} for dn in self.DATASETS},
                   "Majority Resolution": {dn: {e: [] for e in user_labeling_error_range} for dn in self.DATASETS}}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for e in user_labeling_error_range:
                    detector.USER_LABELING_ACCURACY = 1.0 - e
                    detector.LABEL_PROPAGATION_METHOD = "homogeneity"
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Homogeneity Resolution"][dataset_name][e].append(er)
                    detector.LABEL_PROPAGATION_METHOD = "majority"
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results["Majority Resolution"][dataset_name][e].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for propagation_approach in results:
            row = [propagation_approach]
            for dataset_name in self.DATASETS:
                f_list = [numpy.mean(numpy.array(results[propagation_approach][dataset_name][e]), axis=0)[2] for e in user_labeling_error_range]
                row.append(((len(user_labeling_error_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
            table_1.add_row(row)
        print("System effectiveness in the presence of user. (F1 score with the respective user labeling error portions: {})".format(user_labeling_error_range))
        print(table_1)

    def experiment_6(self):
        """
        This method conducts experiment 6.
        """
        print("------------------------------------------------------------------------\n"
              "--------------------Experiment 6: System Scalability--------------------\n"
              "------------------------------------------------------------------------")
        rows_counts_list = [50000, 100000, 150000, 200000]
        results = {rc: [] for rc in rows_counts_list}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        dataset_dictionary = {
            "name": "tax",
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "tax", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "tax", "clean.csv"))
        }
        d_tax = raha.dataset.Dataset(dataset_dictionary)
        for r in range(self.RUN_COUNT):
            for rows_count in rows_counts_list:
                dataset_name = "tax_{}".format(rows_count)
                nd_folder_path = os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name)
                if os.path.exists(nd_folder_path):
                    shutil.rmtree(nd_folder_path)
                os.mkdir(nd_folder_path)
                d_tax.write_csv_dataset(os.path.join(nd_folder_path, "dirty.csv"), d_tax.dataframe.iloc[:rows_count, :])
                d_tax.write_csv_dataset(os.path.join(nd_folder_path, "clean.csv"), d_tax.clean_dataframe.iloc[:rows_count, :])
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                start_time = time.time()
                detection_dictionary = detector.run(dataset_dictionary)
                er = d.get_data_cleaning_evaluation(detection_dictionary)[:3] + [time.time() - start_time]
                results[rows_count].append(er)
                shutil.rmtree(nd_folder_path)
        table_1 = prettytable.PrettyTable(["Rows Count", "F1 Score", "Runtime"])
        for rows_count in rows_counts_list:
            aggregated_list = numpy.mean(numpy.array(results[rows_count]), axis=0)
            row = [rows_count, "{:.2f}".format(aggregated_list[2]), "{:.0f}".format(aggregated_list[3])]
            table_1.add_row(row)
        print("System scalability with respect to the number of rows in tax dataset.")
        print(table_1)

    def experiment_7(self):
        """
        This method conducts experiment 7.
        """
        print("------------------------------------------------------------------------\n"
              "------------------Experiment 7: Feature Impact Analysis-----------------\n"
              "------------------------------------------------------------------------")
        results = {"AdaBoost": {dn: [] for dn in self.DATASETS},
                   "Decision Tree": {dn: [] for dn in self.DATASETS},
                   "Gradient Boosting": {dn: [] for dn in self.DATASETS},
                   "Gaussian Naive Bayes": {dn: [] for dn in self.DATASETS},
                   "Stochastic Gradient Descent": {dn: [] for dn in self.DATASETS},
                   "Support Vectors Machines": {dn: [] for dn in self.DATASETS}}
        detector = raha.detection.Detection()
        detector.VERBOSE = False
        for r in range(self.RUN_COUNT):
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for classification_model in results:
                    if classification_model == "AdaBoost":
                        detector.CLASSIFICATION_MODEL = "ABC"
                    if classification_model == "Decision Tree":
                        detector.CLASSIFICATION_MODEL = "DTC"
                    if classification_model == "Gradient Boosting":
                        detector.CLASSIFICATION_MODEL = "GBC"
                    if classification_model == "Gaussian Naive Bayes":
                        detector.CLASSIFICATION_MODEL = "GNB"
                    if classification_model == "Stochastic Gradient Descent":
                        detector.CLASSIFICATION_MODEL = "SGDC"
                    if classification_model == "Support Vectors Machines":
                        detector.CLASSIFICATION_MODEL = "SVC"
                    detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results[classification_model][dataset_name].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for classification_model in results:
            row = [classification_model]
            for dataset_name in self.DATASETS:
                p, r, f = numpy.mean(numpy.array(results[classification_model][dataset_name]), axis=0)
                row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
            table_1.add_row(row)
        print("System effectiveness with different classification models. (Precision, recall, f1 score)")
        print(table_1)
########################################


########################################
if __name__ == "__main__":
    app = Benchmark()
    if "fast" in sys.argv:
        app.RUN_COUNT = 1
        app.DATASETS = ["hospital", "flights"]
    if "1" in sys.argv:
        app.experiment_1()
    if "2" in sys.argv:
        app.experiment_2()
    if "3" in sys.argv:
        app.experiment_3()
    if "4" in sys.argv:
        app.experiment_4()
    if "5" in sys.argv:
        app.experiment_5()
    if "6" in sys.argv:
        app.experiment_6()
    if "7" in sys.argv:
        app.experiment_7()
########################################





