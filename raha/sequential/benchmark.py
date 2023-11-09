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
import matplotlib.pyplot

import raha
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
        self.DATASETS = ["hospital", "flights", "beers", "rayyan", "movies_1"]

    def experiment_1(self):
        """
        This method conducts experiment 1.
        """
        print("------------------------------------------------------------------------\n"
              "-----------------Experiment 1: Comparison with Baselines----------------\n"
              "------------------------------------------------------------------------")
        stand_alone_systems = ["dBoost", "NADEEF", "KATARA", "ActiveClean", "Raha"]
        results = {sas: {dn: [] for dn in self.DATASETS} for sas in stand_alone_systems}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            competitor = raha.baselines.Baselines()
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for stand_alone_system in stand_alone_systems[::-1]:
                    if stand_alone_system == "dBoost":
                        detection_dictionary = competitor.run_dboost(dataset_dictionary)
                    if stand_alone_system == "NADEEF":
                        detection_dictionary = competitor.run_nadeef(dataset_dictionary)
                    if stand_alone_system == "KATARA":
                        detection_dictionary = competitor.run_katara(dataset_dictionary)
                    if stand_alone_system == "ActiveClean":
                        detection_dictionary = competitor.run_activeclean(dataset_dictionary)
                    if stand_alone_system == "Raha":
                        detection_dictionary = detector.run(dataset_dictionary)
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                    results[stand_alone_system][dataset_name].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for stand_alone_system in stand_alone_systems:
            row = [stand_alone_system]
            for dataset_name in self.DATASETS:
                p, r, f = numpy.mean(numpy.array(results[stand_alone_system][dataset_name]), axis=0)
                row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
            table_1.add_row(row)
        tuple_wise_systems = ["ActiveClean", "Raha"]
        results = {tws: {dn: [] for dn in self.DATASETS} for tws in tuple_wise_systems}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            competitor = raha.baselines.Baselines()
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for tuple_wise_system in tuple_wise_systems:
                    if tuple_wise_system == "ActiveClean":
                        detection_dictionary = competitor.run_activeclean(dataset_dictionary)
                    if tuple_wise_system == "Raha":
                        detection_dictionary = detector.run(dataset_dictionary)
                    er = raha.utilities.get_tuple_wise_evaluation(d, detection_dictionary)
                    results[tuple_wise_system][dataset_name].append(er)
        table_2 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        for tuple_wise_system in tuple_wise_systems:
            row = [tuple_wise_system]
            for dataset_name in self.DATASETS:
                p, r, f = numpy.mean(numpy.array(results[tuple_wise_system][dataset_name]), axis=0)
                row.append("{:.2f}, {:.2f}, {:.2f}".format(p, r, f))
            table_2.add_row(row)
        sampling_range = [20, 40, 60, 80, 100]
        aggregator_systems = ["Min-k", "Maximum Entropy", "Metadata Driven", "Raha"]
        results = {ags: {dn: {s: [] for s in sampling_range} for dn in self.DATASETS} for ags in aggregator_systems}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            competitor = raha.baselines.Baselines()
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for s in sampling_range:
                    for aggregator_system in aggregator_systems:
                        if aggregator_system == "Min-k":
                            detection_dictionary = competitor.run_min_k(dataset_dictionary)
                        if aggregator_system == "Maximum Entropy":
                            detection_dictionary = competitor.run_maximum_entropy(dataset_dictionary, s)
                        if aggregator_system == "Metadata Driven":
                            detection_dictionary = competitor.run_metadata_driven(dataset_dictionary, s)
                        if aggregator_system == "Raha":
                            detector.LABELING_BUDGET = s
                            detection_dictionary = detector.run(dataset_dictionary)
                        er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                        results[aggregator_system][dataset_name][s].append(er)
        table_3 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        f_scores = {}
        for aggregator_system in aggregator_systems:
            row = [aggregator_system]
            for dataset_name in self.DATASETS:
                f_list = [numpy.mean(numpy.array(results[aggregator_system][dataset_name][s]), axis=0)[2] for s in sampling_range]
                row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
                f_scores[(aggregator_system, dataset_name)] = f_list
            table_3.add_row(row)
        fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=len(self.DATASETS))
        for i, ax in enumerate(axs):
            ax.set_title(self.DATASETS[i])
            ax.set(xlabel="Labeled Tuples Count", ylabel="F1 Score")
            ax.set_ylim([0.0, 1.0])
            ax.grid(True)
            for aggregator_system in aggregator_systems:
                f_list = f_scores[(aggregator_system, self.DATASETS[i])]
                ax.plot([0] + sampling_range, [0 if aggregator_system != "Min-k" else f_list[0]] + f_list)
            ax.legend(aggregator_systems, bbox_to_anchor=(0.8, -0.07))
        print("Comparison with the stand-alone error detection tools. (Precision, recall, f1 score)")
        print(table_1)
        print("Comparison in terms of detecting erroneous tuples. (Tuple-wise precision, recall, f1 score)")
        print(table_2)
        print("Comparison with the error detection aggregators. (F1 score with the respective numbers of labeled tuples: {})".format(sampling_range))
        print(table_3)
        fig.suptitle("Comparison with the error detection aggregators.", fontsize=20)
        matplotlib.pyplot.show()

    def experiment_2(self):
        """
        This method conducts experiment 2.
        """
        print("------------------------------------------------------------------------\n"
              "------------------Experiment 2: Feature Impact Analysis-----------------\n"
              "------------------------------------------------------------------------")
        feature_specifications = ["TF-IDF", "All - OD", "All - PVD", "All - RVD", "All - KBVD", "All"]
        results = {fs: {dn: [] for dn in self.DATASETS} for fs in feature_specifications}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for feature_specification in feature_specifications:
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
        for feature_specification in feature_specifications:
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
        sampling_approaches = ["Uniform", "Clustering-Based"]
        results = {sa: {dn: {s: [] for s in sampling_range} for dn in self.DATASETS} for sa in sampling_approaches}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for s in sampling_range:
                    detector.LABELING_BUDGET = s
                    for sampling_approach in sampling_approaches:
                        if sampling_approach == "Uniform":
                            detector.CLUSTERING_BASED_SAMPLING = False
                        if sampling_approach == "Clustering-Based":
                            detector.CLUSTERING_BASED_SAMPLING = True
                        detection_dictionary = detector.run(dataset_dictionary)
                        er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                        results[sampling_approach][dataset_name][s].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        f_scores = {}
        for sampling_approach in sampling_approaches:
            row = [sampling_approach]
            for dataset_name in self.DATASETS:
                f_list = [numpy.mean(numpy.array(results[sampling_approach][dataset_name][s]), axis=0)[2] for s in sampling_range]
                row.append(((len(sampling_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
                f_scores[(sampling_approach, dataset_name)] = f_list
            table_1.add_row(row)
        fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=len(self.DATASETS))
        for i, ax in enumerate(axs):
            ax.set_title(self.DATASETS[i])
            ax.set(xlabel="Labeled Tuples Count", ylabel="F1 Score")
            ax.set_ylim([0.0, 1.0])
            ax.grid(True)
            for sampling_approach in sampling_approaches:
                f_list = f_scores[(sampling_approach, self.DATASETS[i])]
                ax.plot([0] + sampling_range, [0] + f_list)
            ax.legend(sampling_approaches, bbox_to_anchor=(0.8, -0.07))
        print("System effectiveness with different sampling approaches. (F1 score with the respective numbers of labeled tuples: {})".format(sampling_range))
        print(table_1)
        fig.suptitle("System effectiveness with different sampling approaches.", fontsize=20)
        matplotlib.pyplot.show()

    def experiment_4(self):
        """
        This method conducts experiment 4.
        """
        print("------------------------------------------------------------------------\n"
              "------------Experiment 4: Strategy Filtering Impact Analysis------------\n"
              "------------------------------------------------------------------------")
        historical_datasets = ["hospital", "flights", "beers", "rayyan", "movies_1"]
        strategy_filtering_approaches = ["No Strategy Filtering",
                                         "Strategy Filtering via Historical Data",
                                         "Strategy Filtering via Least Effective Selection",
                                         "Strategy Filtering via Uniform Selection",
                                         "Strategy Filtering via Most Effective Selection"]
        results = {sfa: {dn: [] for dn in self.DATASETS} for sfa in strategy_filtering_approaches}
        historical_dataset_dictionaries = []
        for dataset_name in historical_datasets:
            dataset_dictionary = {
                "name": dataset_name,
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
            }
            raha.utilities.dataset_profiler(dataset_dictionary)
            raha.utilities.evaluation_profiler(dataset_dictionary)
            historical_dataset_dictionaries.append(dataset_dictionary)
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for strategy_filtering_approach in strategy_filtering_approaches:
                    if strategy_filtering_approach == "No Strategy Filtering":
                        detector.STRATEGY_FILTERING = False
                        detection_dictionary = detector.run(dataset_dictionary)
                        strategies_count, runtime = raha.utilities.get_strategies_count_and_runtime(dataset_dictionary)
                    elif strategy_filtering_approach == "Strategy Filtering via Historical Data":
                        selected_strategies = raha.utilities.get_selected_strategies_via_historical_data(dataset_dictionary, historical_dataset_dictionaries)
                        detection_dictionary = raha.utilities.error_detection_with_selected_strategies(dataset_dictionary, selected_strategies)
                        strategies_count = len(selected_strategies)
                        filtered_strategies_count = strategies_count
                        runtime = sum([sp["runtime"] for sp in selected_strategies])
                    else:
                        worst_strategies, random_strategies, best_strategies = raha.utilities.get_selected_strategies_via_ground_truth(dataset_dictionary, filtered_strategies_count)
                        if strategy_filtering_approach == "Strategy Filtering via Least Effective Selection":
                            detection_dictionary = raha.utilities.error_detection_with_selected_strategies(dataset_dictionary, worst_strategies)
                            strategies_count = len(worst_strategies)
                            runtime = sum([sp["runtime"] for sp in worst_strategies])
                        if strategy_filtering_approach == "Strategy Filtering via Uniform Selection":
                            detection_dictionary = raha.utilities.error_detection_with_selected_strategies(dataset_dictionary, random_strategies)
                            strategies_count = len(random_strategies)
                            runtime = sum([sp["runtime"] for sp in random_strategies])
                        if strategy_filtering_approach == "Strategy Filtering via Most Effective Selection":
                            detection_dictionary = raha.utilities.error_detection_with_selected_strategies(dataset_dictionary, best_strategies)
                            strategies_count = len(best_strategies)
                            runtime = sum([sp["runtime"] for sp in best_strategies])
                    er = d.get_data_cleaning_evaluation(detection_dictionary)[:3] + [strategies_count, runtime]
                    results[strategy_filtering_approach][dataset_name].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        f_scores_and_runtime = {}
        for strategy_filtering_approach in strategy_filtering_approaches:
            row = [strategy_filtering_approach]
            for dataset_name in self.DATASETS:
                p, r, f, sc, rt = numpy.mean(numpy.array(results[strategy_filtering_approach][dataset_name]), axis=0)
                row.append("{:.2f}, {:.2f}, {:.2f}, {:.0f}, {:.0f}".format(p, r, f, sc, rt))
                f_scores_and_runtime[(strategy_filtering_approach, dataset_name)] = f, rt
            table_1.add_row(row)
        fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=2)
        width = 0.35
        x = numpy.arange(len(self.DATASETS))
        for i, ax in enumerate(axs):
            if i == 0:
                r_1 = [f_scores_and_runtime[("No Strategy Filtering", dn)][1] for dn in self.DATASETS]
                r_2 = [f_scores_and_runtime[("Strategy Filtering via Historical Data", dn)][1] for dn in self.DATASETS]
                ax.bar(x - width / 2, r_1, width)
                ax.bar(x + width / 2, r_2, width)
                ax.set(xlabel="Dataset", ylabel="Runtime (seconds)")
                ax.set_xticks(x)
                ax.set_xticklabels(self.DATASETS)
                ax.set_yscale("log")
                ax.grid(True)
                ax.legend(["No Strategy Filtering", "Strategy Filtering via Historical Data"], bbox_to_anchor=(0.6, -0.07))
            if i == 1:
                for si, strategy_filtering_approach in enumerate(strategy_filtering_approaches):
                    f = [f_scores_and_runtime[(strategy_filtering_approach, dn)][0] for dn in self.DATASETS]
                    ax.bar(x + (si - 3) * width / 5, f, width / 10)
                ax.set(xlabel="Dataset", ylabel="F1 Score")
                ax.set_xticks(x)
                ax.set_xticklabels(self.DATASETS)
                ax.set_ylim([0.0, 1.0])
                ax.grid(True)
                ax.legend(strategy_filtering_approaches, bbox_to_anchor=(0.6, -0.07))
        print("System performance with different strategy filtering approaches. (Precision, recall, f1 score, selected strategies count, and runtime (seconds))")
        print(table_1)
        fig.suptitle("System performance with different strategy filtering approaches.", fontsize=20)
        matplotlib.pyplot.show()

    def experiment_5(self):
        """
        This method conducts experiment 5.
        """
        print("------------------------------------------------------------------------\n"
              "------------Experiment 5: User Labeling Error Impact Analysis-----------\n"
              "------------------------------------------------------------------------")
        user_labeling_error_range = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        label_propagation_approaches = ["Homogeneity Resolution", "Majority Resolution"]
        results = {lpa: {dn: {e: [] for e in user_labeling_error_range} for dn in self.DATASETS} for lpa in label_propagation_approaches}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for e in user_labeling_error_range:
                    detector.USER_LABELING_ACCURACY = 1.0 - e
                    for label_propagation_approach in label_propagation_approaches:
                        if label_propagation_approach == "Homogeneity Resolution":
                            detector.LABEL_PROPAGATION_METHOD = "homogeneity"
                        if label_propagation_approach == "Majority Resolution":
                            detector.LABEL_PROPAGATION_METHOD = "majority"
                        detection_dictionary = detector.run(dataset_dictionary)
                        er = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
                        results[label_propagation_approach][dataset_name][e].append(er)
        table_1 = prettytable.PrettyTable(["Approach"] + self.DATASETS)
        f_scores = {}
        for label_propagation_approach in label_propagation_approaches:
            row = [label_propagation_approach]
            for dataset_name in self.DATASETS:
                f_list = [numpy.mean(numpy.array(results[label_propagation_approach][dataset_name][e]), axis=0)[2] for e in user_labeling_error_range]
                row.append(((len(user_labeling_error_range) - 1) * "{:.2f}, " + "{:.2f}").format(*f_list))
                f_scores[(label_propagation_approach, dataset_name)] = f_list
            table_1.add_row(row)
        fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=len(label_propagation_approaches))
        for i, ax in enumerate(axs):
            ax.set_title(label_propagation_approaches[i])
            ax.set(xlabel="User Labeling Error Rate (%)", ylabel="F1 Score")
            ax.set_ylim([0.0, 1.0])
            ax.grid(True)
            for dataset_name in self.DATASETS:
                f_list = f_scores[(label_propagation_approaches[i], dataset_name)]
                ax.plot([e * 100 for e in user_labeling_error_range], f_list)
            ax.legend(self.DATASETS, bbox_to_anchor=(0.8, -0.07))
        print("System effectiveness in the presence of user. (F1 score with the respective user labeling error portions: {})".format(user_labeling_error_range))
        print(table_1)
        fig.suptitle("System effectiveness in the presence of user.", fontsize=20)
        matplotlib.pyplot.show()

    def experiment_6(self):
        """
        This method conducts experiment 6.
        """
        print("------------------------------------------------------------------------\n"
              "--------------------Experiment 6: System Scalability--------------------\n"
              "------------------------------------------------------------------------")
        rows_counts = [50000, 100000, 150000, 200000]
        results = {rc: [] for rc in rows_counts}
        dataset_dictionary = {
            "name": "tax",
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "tax", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "tax", "clean.csv"))
        }
        d_tax = raha.dataset.Dataset(dataset_dictionary)
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for rows_count in rows_counts:
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
        for rows_count in rows_counts:
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
        classification_models = ["AdaBoost", "Decision Tree", "Gradient Boosting", "Gaussian Naive Bayes",
                                 "Stochastic Gradient Descent", "Support Vectors Machines"]
        results = {cm: {dn: [] for dn in self.DATASETS} for cm in classification_models}
        for r in range(self.RUN_COUNT):
            detector = raha.detection.Detection()
            detector.VERBOSE = False
            for dataset_name in self.DATASETS:
                dataset_dictionary = {
                    "name": dataset_name,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
                }
                d = raha.dataset.Dataset(dataset_dictionary)
                for classification_model in classification_models:
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
        for classification_model in classification_models:
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





