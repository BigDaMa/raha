########################################
# Raha
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
# ########################################


########################################
import os
import sys
import re
import math
import time
import string
import json
import random
import operator
import gzip
import pickle
import shutil
import itertools
import numpy
import pandas
import scipy.stats
import scipy.spatial
import scipy.cluster
import sklearn.cluster
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.neural_network
import sklearn.kernel_ridge
import sklearn.ensemble
import sklearn.feature_extraction
#import IPython.display
#import ipywidgets
import dataset
import data_cleaning_tool
import multiprocessing as mp
import myGlobals
########################################
def run_strategy(tool_and_configurations):
    tool_name = tool_and_configurations[0]
    configuration = tool_and_configurations[1]
    start_time = time.time()
    strategy_name = json.dumps([tool_name, configuration])
    temp_domain_specific_path = myGlobals.d.name + "-temp_ds-" + "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    if tool_name == "katara":
        if not os.path.exists(temp_domain_specific_path):
            os.mkdir(temp_domain_specific_path)
        shutil.copyfile(os.path.join("tools", "KATARA", "dominSpecific", configuration[0]),
                        os.path.join(temp_domain_specific_path, "temp_file.rel.txt"))
        configuration = [temp_domain_specific_path]
    td = {"name": tool_name, "configuration": configuration}
    t = data_cleaning_tool.DataCleaningTool(td)
    try:
        detected_cells_list = t.run(myGlobals.d).keys()
    except:
        sys.stderr.write("I cannot run the error detection tool!\n")
        detected_cells_list = []
    strategy_profile = {
        "name": strategy_name,
        "output": detected_cells_list,
        "runtime": time.time() - start_time
    }
    print "Running {} is done. Output size = {}".format(strategy_name, len(detected_cells_list))
    if tool_name == "katara":
        if os.path.exists(temp_domain_specific_path):
            shutil.rmtree(temp_domain_specific_path)
    return [strategy_profile, tool_name]

def extract_features(args):
    j = args[0]
    attribute = args[1]

    print "Extracting features for column {}...".format(j)
    o_fv = {(i, j): [] for i in range(myGlobals.d.dataframe.shape[0])}
    p_fv = {(i, j): [] for i in range(myGlobals.d.dataframe.shape[0])}
    r_fv = {(i, j): [] for i in range(myGlobals.d.dataframe.shape[0])}
    k_fv = {(i, j): [] for i in range(myGlobals.d.dataframe.shape[0])}
    for s in myGlobals.all_strategies:
        for i in range(myGlobals.d.dataframe.shape[0]):
            cell = (i, j)
            b = 1 if s in myGlobals.cells_strategies[cell] else 0
            if "dboost" in s:
                o_fv[cell].append(b)
            if "regex" in s:
                p_fv[cell].append(b)
            if "fd_checker" in s:
                r_fv[cell].append(b)
            if "katara" in s:
                k_fv[cell].append(b)

    def noise_features_remover(fv_dic):
        x_data = numpy.array(fv_dic.values(), dtype=numpy.float)
        non_identical_columns = numpy.any(x_data != x_data[0, :], axis=0)
        x_data = x_data[:, non_identical_columns]
        for index, cell in enumerate(fv_dic):
            fv_dic[cell] = x_data[index].tolist()
        return fv_dic

    o_fv = noise_features_remover(o_fv)
    p_fv = noise_features_remover(p_fv)
    r_fv = noise_features_remover(r_fv)
    k_fv = noise_features_remover(k_fv)

    return [o_fv, p_fv, r_fv, k_fv, attribute]


########################################
class Raha:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor does nothing.
        """
        self.RESULTS_FOLDER = "results"
        self.DATASETS = {
            # "hospital": {
            #     "name": "hospital",
            #     "path": os.path.join(self.DATASETS_FOLDER, "hospital", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "hospital", "clean.csv")
            # },
            # "flights": {
            #     "name": "flights",
            #     "path": os.path.join(self.DATASETS_FOLDER, "flights", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "flights", "clean.csv")
            # },
            # "address": {
            #     "name": "address",
            #     "path": os.path.join(self.DATASETS_FOLDER, "address", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "address", "clean.csv")
            # },
            # "beers": {
            #     "name": "beers",
            #     "path": os.path.join(self.DATASETS_FOLDER, "beers", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "beers", "clean.csv")
            # },
            # "rayyan": {
            #     "name": "rayyan",
            #     "path": os.path.join(self.DATASETS_FOLDER, "rayyan", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "rayyan", "clean.csv")
            # },
            # "movies_1": {
            #     "name": "movies_1",
            #     "path": os.path.join(self.DATASETS_FOLDER, "movies_1", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "movies_1", "clean.csv")
            # },
            # "merck": {
            #     "name": "merck",
            #     "path": os.path.join(self.DATASETS_FOLDER, "merck", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "merck", "clean.csv")
            # },
            # # ----------------------------------------------------
            # "toy": {
            #     "name": "toy",
            #     "path": os.path.join(self.DATASETS_FOLDER, "toy", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "toy", "clean.csv")
            # },
            # "restaurants_4": {
            #     "name": "restaurants_4",
            #     "path": os.path.join(self.DATASETS_FOLDER, "restaurants_4", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "restaurants_4", "clean.csv")
            # },
            # "salaries": {
            #     "name": "salaries",
            #     "path": os.path.join(self.DATASETS_FOLDER, "salaries", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "salaries", "clean.csv")
            # },
            # "soccer": {
            #     "name": "soccer",
            #     "path": os.path.join(self.DATASETS_FOLDER, "soccer", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "soccer", "clean.csv")
            # },
            # "tax": {
            #     "name": "tax",
            #     "path": os.path.join(self.DATASETS_FOLDER, "tax", "dirty.csv"),
            #     "clean_path": os.path.join(self.DATASETS_FOLDER, "tax", "clean.csv")
            # }
        }
        self.ERROR_DETECTION_TOOLS = ["dboost", "regex", "fd_checker", "katara"]  # ["dboost", "regex", "fd_checker", "katara"]
        self.RUN_COUNT = 1  # [1, 10, 20]
        self.LABELING_BUDGET = 20
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.STRATEGY_FILTERING = False  # [False, True]   # Uncomment all the datasets for selecting "True"!
        self.BASELINES = ["dBoost"]   # ["dBoost", "NADEEF", "KATARA", "Min-k", "Maximum Entropy", "Metadata Driven", "ActiveClean"]

    def strategy_profiler(self, d):
        """
        This method runs all the error detections strategies with all the possible configurations on the dataset.
        """
        if not os.path.exists(os.path.join(self.RESULTS_FOLDER, d.name)):
            os.mkdir(os.path.join(self.RESULTS_FOLDER, d.name))
        sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-profiling")
        if os.path.exists(sp_folder_path):
            sys.stderr.write("The error detection strategies have already run on this dataset!\n")
            return
        else:
            os.mkdir(sp_folder_path)
            tool_and_configurations = []
            for tool_name in self.ERROR_DETECTION_TOOLS:
                configuration_list = []
                if tool_name == "dboost":
                    configuration_list = [list(a) for a in
                                          list(itertools.product(["histogram"], ["0.7", "0.8", "0.9"], ["0.1", "0.2", "0.3"])) +
                                          list(itertools.product(["gaussian"], ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"])) +
                                          list(itertools.product(["mixture"], ["3", "5", "8"], ["0.01", "0.10", "0.20", "0.30"])) +
                                          list(itertools.product(["partitionedhistogram"], ["3", "5", "8"], ["0.7", "0.8", "0.9"], ["0.1", "0.2", "0.3"]))]
                elif tool_name == "regex":
                    for attribute in d.dataframe.columns:
                        column_data = "".join(d.dataframe[attribute].tolist())
                        characters_dictionary = {ch: 1 for ch in column_data}
                        for ch in characters_dictionary:
                            configuration_list.append([[attribute, "[" + ch + "]", "OM"]])
                elif tool_name == "fd_checker":
                    al = d.dataframe.columns.tolist()
                    configuration_list = [[[a, b]] for (a, b) in itertools.product(al, al) if a != b]
                elif tool_name == "katara":
                    configuration_list = [[p] for p in os.listdir("tools/KATARA/dominSpecific")]

                tool_and_configurations.extend([[tool_name, i] for i in configuration_list])

            pool = mp.Pool()
            strategy_profiles = pool.map(run_strategy, tool_and_configurations)

            for strategy_profile_list in strategy_profiles:
                tool_name = strategy_profile_list[1]
                strategy_profile = strategy_profile_list[0]
                pickle.dump(strategy_profile, open(os.path.join(sp_folder_path, tool_name + "-" + str(
                    len(os.listdir(sp_folder_path))) + ".dictionary"), "wb"))

    def feature_generator(self, d):
        """
        This method generates a feature vector for each data cell.
        """
        fv_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "feature-vectors")
        sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-profiling")
        if self.STRATEGY_FILTERING:
            if not os.path.exists(os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "strategy-profiling")):
                self.strategy_filterer(d)
            fv_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "feature-vectors")
            sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "strategy-profiling")
        if not os.path.exists(fv_folder_path):
            os.mkdir(fv_folder_path)
        myGlobals.all_strategies = {}
        myGlobals.cells_strategies = {cell: {} for cell in itertools.product(range(d.dataframe.shape[0]), range(d.dataframe.shape[1]))}
        for strategy_file in os.listdir(sp_folder_path):
            strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
            myGlobals.all_strategies[strategy_profile["name"]] = 1
            for cell in strategy_profile["output"]:
                myGlobals.cells_strategies[cell][strategy_profile["name"]] = 1

        mp_args = [[j,attribute] for j, attribute in enumerate(d.dataframe.columns.tolist())]

        pool = mp.Pool()
        features = pool.map(extract_features, mp_args)

        for feature in features:
            o_fv = feature[0]
            p_fv = feature[1]
            r_fv = feature[2]
            k_fv = feature[3]
            attribute = feature[4]
            pickle.dump([o_fv, p_fv, r_fv, k_fv], gzip.open(os.path.join(fv_folder_path, attribute + ".dictionary"), "wb"))

    def error_detector(self, d):
        """
        This method cluster data cells and asks user to label them. Next, it trains a classifier per data column.
        """
        ed_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "error-detection")
        fv_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "feature-vectors")
        if self.STRATEGY_FILTERING:
            ed_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "error-detection")
            fv_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "feature-vectors")
        if not os.path.exists(ed_folder_path):
            os.mkdir(ed_folder_path)
        fv = {j: {(i, j): [] for i in range(d.dataframe.shape[0])} for j in range(d.dataframe.shape[1])}
        sampling_range = range(1, self.LABELING_BUDGET + 1)
        clustering_range = range(2, self.LABELING_BUDGET + 2)
        clusters_k_j_c_ce = {k: {j: {} for j in range(d.dataframe.shape[1])} for k in clustering_range}
        cells_clusters_k_j_ce = {k: {j: {} for j in range(d.dataframe.shape[1])} for k in clustering_range}
        # clusters_center_k_jc = {k: {j: {} for j in range(d.columns_count)} for k in clustering_range}
        for j, attribute in enumerate(d.dataframe.columns.tolist()):
            print "Building clustering model for column {}...".format(j)
            o_fv, p_fv, r_fv, k_fv = pickle.load(gzip.open(os.path.join(fv_folder_path, attribute + ".dictionary"), "rb"))
            for i in range(d.dataframe.shape[0]):
                if "dboost" in self.ERROR_DETECTION_TOOLS:
                    fv[j][(i, j)] += o_fv[(i, j)]
                if "regex" in self.ERROR_DETECTION_TOOLS:
                    fv[j][(i, j)] += p_fv[(i, j)]
                if "fd_checker" in self.ERROR_DETECTION_TOOLS:
                    fv[j][(i, j)] += r_fv[(i, j)]
                if "katara" in self.ERROR_DETECTION_TOOLS:
                    fv[j][(i, j)] += k_fv[(i, j)]
            # # Baseline: TF-IDF features same as ActiveClean
            # vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
            # text = [row[j] for row in d.table[1:]]
            # try:
            #     acfv = vectorizer.fit_transform(text).toarray()
            # except:
            #     acfv = [] * d.rows_count
            # for i, afv in enumerate(acfv):
            #     fv[j][(i + 1, j)] = afv
            try:
                clustering_model = scipy.cluster.hierarchy.linkage(numpy.array(fv[j].values()), method="average", metric="cosine")
            except:
                continue
            for k in clusters_k_j_c_ce:
                model_labels = [l - 1 for l in scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                for index, c in enumerate(model_labels):
                    if c not in clusters_k_j_c_ce[k][j]:
                        clusters_k_j_c_ce[k][j][c] = {}
                    cell = fv[j].keys()[index]
                    clusters_k_j_c_ce[k][j][c][cell] = list(fv[j][cell])
                    cells_clusters_k_j_ce[k][j][cell] = c
                # for c in clusters_k_j_c_ce[k][j]:
                #     if len(clusters_k_j_c_ce[k][j][c]) > 0:
                #         pairwise_distance = sklearn.metrics.pairwise.pairwise_distances(clusters_k_j_c_ce[k][j][c].values(), metric="euclidean")
                #         clusters_center_k_jc[k][(j, c)] = clusters_k_j_c_ce[k][j][c].values()[numpy.argmin(pairwise_distance.sum(axis=0))]
        aggregate_results = {s: [] for s in sampling_range}
        for r in range(self.RUN_COUNT):
            print "Run {}...".format(r)
            labeled_tuples = {}
            labeled_cells = {}
            for k in clusters_k_j_c_ce:
                labels_per_cluster = {}
                for j in range(d.dataframe.shape[1]):
                    for c in clusters_k_j_c_ce[k][j]:
                        labels_per_cluster[(j, c)] = {cell: labeled_cells[cell]
                                                      for cell in clusters_k_j_c_ce[k][j][c] if cell[0] in labeled_tuples}
                tuple_score = {i: 0.0 for i in range(d.dataframe.shape[0]) if i not in labeled_tuples}
                for i in tuple_score:
                    score = 0.0
                    for j in range(d.dataframe.shape[1]):
                        if not clusters_k_j_c_ce[k][j]:
                            continue
                        cell = (i, j)
                        c = cells_clusters_k_j_ce[k][j][cell]
                        cell_score = 1.0
                        cell_score *= math.exp(-len(labels_per_cluster[(j, c)]))
                        # cell_score *= int(len(labels_per_cluster[(j, c)]) == 0)
                        # cell_score *= min(100, float(d.rows_count) / (k * float(len(clusters_k_j_c_ce[k][j][c]))))
                        # cell_score *= float(sum(fv[j][cell])) / len(fv[j][cell])
                        #     dist = scipy.spatial.distance.euclidean(fv[j][cell], clusters_center_k_jc[k][(j, c)]) / (
                        #             numpy.linalg.norm(fv[j][cell]) * numpy.linalg.norm(clusters_center_k_jc[k][(j, c)]))
                        #     cell_score *= 1.0 - dist
                        score += cell_score
                    tuple_score[i] = math.exp(score)
                sum_tuple_score = sum(tuple_score.values())
                p_tuple_score = [float(v) / sum_tuple_score for v in tuple_score.values()]
                si = numpy.random.choice(tuple_score.keys(), 1, p=p_tuple_score)[0]
                # si, score = max(tuple_score.iteritems(), key=operator.itemgetter(1))
                labeled_tuples[si] = tuple_score[si]
                if hasattr(d, "actual_errors_dictionary"):
                    for j in range(d.dataframe.shape[1]):
                        cell = (si, j)
                        labeled_cells[cell] = int(cell in d.actual_errors_dictionary)
                        if cell in cells_clusters_k_j_ce[k][j]:
                            c = cells_clusters_k_j_ce[k][j][cell]
                            labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
                else:
                    print "Label the dirty cells in the following sampled tuple."
                    sampled_tuple = pandas.DataFrame(data=[d.dataframe.iloc[si, :]], columns=d.dataframe.columns)
                    #IPython.display.display(sampled_tuple)
                    for j in range(d.dataframe.shape[1]):
                        cell = (si, j)
                        value = d.dataframe.iloc[cell]
                        labeled_cells[cell] = int(raw_input("Is the value '{}' dirty?\nType 1 for yes.\nType 0 for no.\n".format(value)))
                        if cell in cells_clusters_k_j_ce[k][j]:
                            c = cells_clusters_k_j_ce[k][j][cell]
                            labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
                extended_labeled_cells = dict(labeled_cells)
                for j in clusters_k_j_c_ce[k]:
                    for c in clusters_k_j_c_ce[k][j]:
                        if len(labels_per_cluster[(j, c)]) > 0 and \
                                sum(labels_per_cluster[(j, c)].values()) in [0, len(labels_per_cluster[(j, c)])]:
                            for cell in clusters_k_j_c_ce[k][j][c]:
                                extended_labeled_cells[cell] = labels_per_cluster[(j, c)].values()[0]
                # # For experiment on user labels accuracy
                # for j in range(d.dataframe.shape[1]):
                #     ran = random.random()
                #     if ran <= USER_ACCURACY:
                #         labeled_cells[(si, j)] = int((si, j) in d.actual_errors_dictionary)
                #     else:
                #         labeled_cells[(si, j)] = 1 - int((si, j) in d.actual_errors_dictionary)
                # for j in range(d.dataframe.shape[1]):
                #     cell = (si, j)
                #     if cell in cells_clusters_k_j_ce:
                #         c = cells_clusters_k_j_ce[k][j][cell]
                #         labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
                # extended_labeled_cells = dict(labeled_cells)
                # for j in clusters_k_j_c_ce[k]:
                #     for c in clusters_k_j_c_ce[k][j]:
                #         if len(labels_per_cluster[(j, c)]) > 0:
                #            l = int(round(sum(labels_per_cluster[(j, c)].values()) / float(len(labels_per_cluster[(j, c)]))))
                #            for cell in clusters_k_j_c_ce[k][j][c]:
                #               extended_labeled_cells[cell] = l
                correction_dictionary = {}
                for j in range(d.dataframe.shape[1]):
                    x_train = [fv[j][(i, j)] for i in range(d.dataframe.shape[0]) if (i, j) in extended_labeled_cells]
                    y_train = [extended_labeled_cells[(i, j)] for i in range(d.dataframe.shape[0]) if (i, j) in extended_labeled_cells]
                    x_test = [fv[j][(i, j)] for i in range(d.dataframe.shape[0])]
                    test_cells = [(i, j) for i in range(d.dataframe.shape[0])]
                    if sum(y_train) == len(y_train):
                        predicted_labels = len(test_cells) * [1]
                    elif sum(y_train) == 0 or len(x_train[0]) == 0:
                        predicted_labels = len(test_cells) * [0]
                    else:
                        if self.CLASSIFICATION_MODEL == "ABC":
                            classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                        if self.CLASSIFICATION_MODEL == "DTC":
                            classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                        if self.CLASSIFICATION_MODEL == "GBC":
                            classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                        if self.CLASSIFICATION_MODEL == "GNB":
                            classification_model = sklearn.naive_bayes.GaussianNB()
                        if self.CLASSIFICATION_MODEL == "KNC":
                            classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                        if self.CLASSIFICATION_MODEL == "SGDC":
                            classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                        if self.CLASSIFICATION_MODEL == "SVC":
                            classification_model = sklearn.svm.SVC(kernel="sigmoid")
                        classification_model.fit(x_train, y_train)
                        predicted_labels = classification_model.predict(x_test)
                    for index, pl in enumerate(predicted_labels):
                        cell = test_cells[index]
                        if (cell[0] in labeled_tuples and extended_labeled_cells[cell]) or \
                                (cell[0] not in labeled_tuples and pl):
                            correction_dictionary[cell] = "JUST A DUMMY VALUE"
                if hasattr(d, "actual_errors_dictionary"):
                    s = len(labeled_tuples)
                    er = d.evaluate_data_cleaning(correction_dictionary)[:3]
                    aggregate_results[s].append(er)
                    # # Tuple-wise evaluation
                    # actual_dirty_tuples = {i: 1 for i in range(d.dataframe.shape[0]) if
                    #                        int(sum([(i, j) in d.actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0)}
                    # tp = 0.0
                    # outputted_tuples = {}
                    # for i, j in correction_dictionary:
                    #     if i not in outputted_tuples:
                    #         outputted_tuples[i] = 1
                    #         if i in actual_dirty_tuples:
                    #             tp += 1.0
                    # p = 0.0 if len(outputted_tuples) == 0 else tp / len(outputted_tuples)
                    # r = 0.0 if len(actual_dirty_tuples) == 0 else tp / len(actual_dirty_tuples)
                    # f = 0.0 if (p + r) == 0.0 else (2 * p * r) / (p + r)
                    # aggregate_results[s].append([p, r, f])
                #else:
                    # pickle.dump(correction_dictionary, open(os.path.join(ed_folder_path, "results.dictionary"), "wb"))
                    #IPython.display.display(d.dataframe.style.apply(
                    #    lambda x: ["background-color: red" if (i, d.dataframe.columns.get_loc(x.name)) in correction_dictionary else ""
                     #              for i, cv in enumerate(x)]))
                if not hasattr(d, "actual_errors_dictionary"):
                    continue_flag = int(raw_input("Would you like to label one more tuple?\nType 1 for yes.\nType 0 for no.\n"))
                    if not continue_flag:
                        break
        if hasattr(d, "actual_errors_dictionary"):
            results_string = "\\addplot[error bars/.cd,y dir=both,y explicit] coordinates{(0,0.0)"
            for s in sampling_range:
                mean = numpy.mean(numpy.array(aggregate_results[s]), axis=0)
                std = numpy.std(numpy.array(aggregate_results[s]), axis=0)
                print "Raha on", d.name
                print "Labeled Tuples Count = {}".format(s)
                print "Precision = {:.2f} +- {:.2f}".format(mean[0], std[0])
                print "Recall = {:.2f} +- {:.2f}".format(mean[1], std[1])
                print "F1 = {:.2f} +- {:.2f}".format(mean[2], std[2])
                print "--------------------"
                results_string += "({},{:.2f})+-(0,{:.2f})".format(s, mean[2], std[2])
            results_string += "}; \\addlegendentry{Ours}"
            print results_string

    def dataset_profiler(self, d):
        """
        This method profiles the columns of dataset.
        """
        if not os.path.exists(os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering")):
            os.mkdir(os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering"))
        dp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "dataset-profiling")
        if not os.path.exists(dp_folder_path):
            os.mkdir(dp_folder_path)
        for attribute in d.dataframe.columns.tolist():
            characters_dictionary = {}
            values_dictionary = {}
            for value in d.dataframe[attribute]:
                for character in list(set(list(value))):
                    if character not in characters_dictionary:
                        characters_dictionary[character] = 0.0
                    characters_dictionary[character] += 1.0
                # for term in list(set(nltk.word_tokenize(value) + [value])):
                if value not in values_dictionary:
                    values_dictionary[value] = 0.0
                values_dictionary[value] += 1.0
            column_profile = {
                "characters": {ch: characters_dictionary[ch] / d.dataframe.shape[0] for ch in characters_dictionary},
                "values": {v: values_dictionary[v] / d.dataframe.shape[0] for v in values_dictionary},
            }
            pickle.dump(column_profile, open(os.path.join(dp_folder_path, attribute + ".dictionary"), "wb"))
        print "The {} dataset is profiled.".format(d.name)

    def evaluation_profiler(self, d):
        """
        This method computes the performance of the error detection strategies on historical data.
        """
        ep_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "evaluation-profiling")
        if not os.path.exists(ep_folder_path):
            os.mkdir(ep_folder_path)
        sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-profiling")
        columns_performance = {j: {} for j in range(d.dataframe.shape[1])}
        strategies_file_list = os.listdir(sp_folder_path)
        for strategy_file in strategies_file_list:
            strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
            strategy_name = strategy_profile["name"]
            strategy_output = strategy_profile["output"]
            for column_index, attribute in enumerate(d.dataframe.columns.tolist()):
                actual_column_errors = {(i, j): 1 for (i, j) in d.actual_errors_dictionary if j == column_index}
                detected_column_cells = [(i, j) for (i, j) in strategy_output if j == column_index]
                tp = 0.0
                for cell in detected_column_cells:
                    if cell in actual_column_errors:
                        tp += 1
                if tp == 0.0:
                    precision = recall = f1 = 0.0
                else:
                    precision = tp / len(detected_column_cells)
                    recall = tp / len(actual_column_errors)
                    f1 = (2 * precision * recall) / (precision + recall)
                columns_performance[column_index][strategy_name] = [precision, recall, f1]
                # if f1 > 0.5:
                #     print "Performance of {} on {} = {:.2f}, {:.2f}, {:.2f}".format(strategy_name, attribute, precision, recall, f1)
        for j, attribute in enumerate(d.dataframe.columns.tolist()):
            pickle.dump(columns_performance[j], open(os.path.join(ep_folder_path, attribute + ".dictionary"), "wb"))
        print "{} error detection strategies are evaluated.".format(len(strategies_file_list))

    def strategy_filterer(self, d):
        """
        This method uses historical data to rank error detection strategies for the dataset and select the top-ranked.
        """
        nsp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "strategy-profiling")
        if not os.path.exists(nsp_folder_path):
            os.mkdir(nsp_folder_path)
        columns_similarity = {}
        for nci, na in enumerate(d.dataframe.columns.tolist()):
            ndp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-filtering", "dataset-profiling")
            ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
            for hdn in self.DATASETS:
                if hdn != d.name:
                    hd = dataset.Dataset(self.DATASETS[hdn])
                    for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                        hdp_folder_path = os.path.join(self.RESULTS_FOLDER, hd.name, "strategy-filtering", "dataset-profiling")
                        hcp = pickle.load(open(os.path.join(hdp_folder_path, ha + ".dictionary"), "rb"))
                        nfv = []
                        hfv = []
                        for k in list(set(ncp["characters"]) | set(hcp["characters"])):
                            nfv.append(ncp["characters"][k]) if k in ncp["characters"] else nfv.append(0.0)
                            hfv.append(hcp["characters"][k]) if k in hcp["characters"] else hfv.append(0.0)
                        for k in list(set(ncp["values"]) | set(hcp["values"])):
                            nfv.append(ncp["values"][k]) if k in ncp["values"] else nfv.append(0.0)
                            hfv.append(hcp["values"][k]) if k in hcp["values"] else hfv.append(0.0)
                        similarity = 1.0 - scipy.spatial.distance.cosine(nfv, hfv)
                        columns_similarity[(d.name, na, hd.name, ha)] = similarity
        print "Column profile similarities are calculated."
        f1_measure = {}
        for hdn in self.DATASETS:
            if hdn != d.name:
                hd = dataset.Dataset(self.DATASETS[hdn])
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    ep_folder_path = os.path.join(self.RESULTS_FOLDER, hd.name, "strategy-filtering", "evaluation-profiling")
                    strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                    if (hd.name, ha) not in f1_measure:
                        f1_measure[(hd.name, ha)] = {}
                    for strategy_name in strategies_performance:
                        f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
        print "Previous strategy performances are loaded."
        strategies_score = {a: {} for a in d.dataframe.columns.tolist()}
        for nci, na in enumerate(d.dataframe.columns.tolist()):
            for hdn in self.DATASETS:
                if hdn != d.name:
                    hd = dataset.Dataset(self.DATASETS[hdn])
                    for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                        similarity = columns_similarity[(d.name, na, hd.name, ha)]
                        if similarity == 0:
                            continue
                        for strategy_name in f1_measure[(hd.name, ha)]:
                            score = similarity * f1_measure[(hd.name, ha)][strategy_name]
                            if score <= 0.0:
                                continue
                            sn = json.loads(strategy_name)
                            if sn[0] == "dboost" or sn[0] == "katara":
                                if strategy_name not in strategies_score[na] or score >= strategies_score[na][strategy_name]:
                                    strategies_score[na][strategy_name] = score
                            elif sn[0] == "regex":
                                sn[1][0][0] = na
                                if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                    strategies_score[na][json.dumps(sn)] = score
                            elif sn[0] == "fd_checker":
                                this_a_i = sn[1][0].index(ha)
                                that_a = sn[1][0][1 - this_a_i]
                                most_similar_a = d.dataframe.columns.tolist()[0]
                                most_similar_v = -1
                                for aa in d.dataframe.columns.tolist():
                                    if aa != na and columns_similarity[(d.name, aa, hd.name, that_a)] > most_similar_v:
                                        most_similar_v = columns_similarity[(d.name, aa, hd.name, that_a)]
                                        most_similar_a = aa
                                sn[1][0][this_a_i] = na
                                sn[1][0][1 - this_a_i] = most_similar_a
                                if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                    strategies_score[na][json.dumps(sn)] = score
                            else:
                                sys.stderr.write("I do not know this error detection tool!\n")
        print "Strategy scores are calculated."
        sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-profiling")
        strategies_output = {}
        strategies_runtime = {}
        for strategy_file in os.listdir(sp_folder_path):
            strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
            strategies_output[strategy_profile["name"]] = strategy_profile["output"]
            strategies_runtime[strategy_profile["name"]] = strategy_profile["runtime"]
        print "Outputs of the strategies are loaded."
        for a in d.dataframe.columns.tolist():
            sorted_strategies = sorted(strategies_score[a].items(), key=operator.itemgetter(1), reverse=True)
            good_strategies = {}
            previous_score = 0.0
            for sn, ss in sorted_strategies:
                if sn not in strategies_output:
                    continue
                first_sum = sum(good_strategies.values())
                second_sum = sum([math.fabs(good_strategies[s_1] - good_strategies[s_2]) for s_1, s_2 in
                                  itertools.product(good_strategies.keys(), good_strategies.keys()) if s_1 > s_2])
                score = first_sum - second_sum
                if score < previous_score:
                    break
                previous_score = score
                good_strategies[sn] = ss
            for sn in good_strategies:
                snd = json.loads(sn)
                runtime = 0.0
                if snd[0] == "dboost" or snd[0] == "katara":
                    runtime = strategies_runtime[sn] / d.dataframe.shape[1]
                elif snd[0] == "regex":
                    runtime = strategies_runtime[sn]
                elif snd[0] == "fd_checker":
                    runtime = strategies_runtime[sn] / 2
                else:
                    sys.stderr.write("I do not know this error detection tool!\n")
                strategy_profile = {
                    "name": sn,
                    "output": [cell for cell in strategies_output[sn] if d.dataframe.columns.tolist()[cell[1]] == a],
                    "runtime": runtime
                }
                pickle.dump(strategy_profile, open(os.path.join(nsp_folder_path, snd[0] + "-" + str(len(os.listdir(nsp_folder_path))) + ".dictionary"), "wb"))
        print "Promising error detection strategies are stored."

    def baselines(self, d):
        """
        This methods implements the baselines.
        """
        b_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "baselines")
        if not os.path.exists(b_folder_path):
            os.mkdir(b_folder_path)
        sp_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "strategy-profiling")
        strategies_file_list = os.listdir(sp_folder_path)
        strategies_output = {}
        for strategy_file in strategies_file_list:
            strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
            strategies_output[strategy_profile["name"]] = strategy_profile["output"]
        dataset_constraints = {
            "hospital": {
                "functions": [["city", "zip"], ["city", "county"], ["zip", "city"], ["zip", "state"], ["zip", "county"],
                              ["county", "state"]],
                "patterns": [["index", "^[\d]+$", "ONM"], ["provider_number", "^[\d]+$", "ONM"],
                             ["zip", "^[\d]{5}$", "ONM"], ["state", "^[a-z]{2}$", "ONM"], ["phone", "^[\d]+$", "ONM"]]
            },
            "flights": {
                "functions": [["flight", "act_dep_time"], ["flight", "sched_arr_time"], ["flight", "act_arr_time"],
                              ["flight", "sched_dep_time"]],
                "patterns": []
            },
            "address": {
                "functions": [["address", "state"], ["address", "zip"], ["zip", "state"]],
                "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["zip", "^[\d]+$", "ONM"], ["ssn", "^[\d]*$", "ONM"]]
            },
            "beers": {
                "functions": [["brewery_id", "brewery_name"], ["brewery_id", "city"], ["brewery_id", "state"]],
                "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["brewery_id", "^[\d]+$", "ONM"]]
            },
            "citation": {
                "functions": [["jounral_abbreviation", "journal_title"], ["jounral_abbreviation", "journal_issn"],
                              ["journal_issn", "journal_title"]],
                "patterns": [
                    ["article_jvolumn", "^$", "OM"],
                    ["article_jissue", "^$", "OM"],
                    ["article_jcreated_at", "^[\d]+[/][\d]+[/][\d]+$|^$", "OM"],
                    ["journal_issn", "^$", "OM"],
                    ["journal_title", "^$", "OM"],
                    ["article_language", "^$", "OM"],
                    ["article_title", "^$", "OM"],
                    ["jounral_abbreviation", "^$", "OM"],
                    ["article_pagination", "^$", "OM"],
                    ["author_list", "^$", "OM"],
                    # ["journal_issn", "^[A-Z][a-z]{2}[-][012][\d]$", "OM"],
                    # ["article_title", """^[A-Za-z_\d [\]<>!?:;./,*()+&'"%-]+$""", "ONM"],
                    # ["journal_title", """^[A-Za-z_\d [\]=:;./,()&'-]+$|^$""", "ONM"],
                    # ["author_list", """^[A-Za-z_\d [\]:./,}{@()&'-]+$""", "ONM"]
                ]
            },
            "movies_1": {
                "functions": [],
                "patterns": [["id", "^tt[\d]+$", "ONM"], ["year", "^[\d]{4}$", "ONM"], ["rating_value", "^[\d.]*$", "ONM"],
                             ["rating_count", "^[\d]*$", "ONM"], ["duration", "^([\d]+[ ]min)*$", "ONM"]]
            },
            "merck": {
                "functions": [],
                "patterns": [["support_level", "^$", "OM"], ["app_status", "^$", "OM"], ["curr_status", "^$", "OM"],
                             ["tower", "^$", "OM"], ["end_users", "^$", "OM"], ["account_manager", "^$", "OM"],
                             ["decomm_dt", "^$", "OM"], ["decomm_start", "^$", "OM"], ["decomm_end", "^$", "OM"],
                             ["end_users", "^(0)$", "OM"],
                             ["retirement", "^(2010|2011|2012|2013|2014|2015|2016|2017|2018)$", "ONM"],
                             ["emp_dta", "^(n|N|y|Y|n/a|N/A|n/A|N/a)$", "ONM"],
                             ["retire_plan", "^(true|True|TRUE|false|False|FALSE|n/a|N/A|n/A|N/a)$", "ONM"],
                             ["bus_import", "^(important|n/a|IP Strategy)$", "OM"],
                             ["division", "^(Merck Research Laboratories|Merck Consumer Health Care)$", "OM"]]
            }
        }

        if "dBoost" in self.BASELINES:
            dboost_configuration = ""
            dboost_performance = -1.0
            random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
            labeled_tuples = {i: 1 for i in random_tuples_list[:int(d.dataframe.shape[0] / 100.0)]}
            for strategy_name in strategies_output:
                tool_name = json.loads(strategy_name)[0]
                if tool_name == "dboost":
                    temp_cd = {}
                    for cell in strategies_output[strategy_name]:
                        temp_cd[cell] = "JUST A DUMMY VALUE"
                    temp_er = d.evaluate_data_cleaning(temp_cd, sampled_rows_dictionary=labeled_tuples)[:3]
                    if temp_er[2] > dboost_performance:
                        dboost_performance = temp_er[2]
                        dboost_configuration = strategy_name
            dboost_correction = {cell: "JUST A DUMMY VALUE" for cell in strategies_output[dboost_configuration]}
            er = d.evaluate_data_cleaning(dboost_correction)[:3]
            pickle.dump(dboost_correction.keys(), open(os.path.join(b_folder_path, "dboost_output.list"), "wb"))
            print "dBoost on", d.name, dboost_correction
            print "Precision = {:.2f}".format(er[0])
            print "Recall = {:.2f}".format(er[1])
            print "F1 = {:.2f}".format(er[2])
            print "--------------------"
        if "NADEEF" in self.BASELINES:
            td = {"name": "nadeef", "configuration": dataset_constraints[d.name]["functions"]}
            t = data_cleaning_tool.DataCleaningTool(td)
            detected_cells_dictionary = t.run(d)
            td = {"name": "regex", "configuration": dataset_constraints[d.name]["patterns"]}
            t = data_cleaning_tool.DataCleaningTool(td)
            detected_cells_dictionary.update(t.run(d))
            nadeef_correction = {cell: "JUST A DUMMY VALUE" for cell in detected_cells_dictionary}
            er = d.evaluate_data_cleaning(nadeef_correction)[:3]
            pickle.dump(nadeef_correction.keys(), open(os.path.join(b_folder_path, "nadeef_output.list"), "wb"))
            print "NADEEF on", d.name
            print "Precision = {:.2f}".format(er[0])
            print "Recall = {:.2f}".format(er[1])
            print "F1 = {:.2f}".format(er[2])
            print "--------------------"
        if "KATARA" in self.BASELINES:
            katara_correction = {}
            for strategy_name in strategies_output:
                tool_name = json.loads(strategy_name)[0]
                if tool_name == "katara":
                    katara_correction.update({cell: "JUST A DUMMY VALUE" for cell in strategies_output[strategy_name]})
            er = d.evaluate_data_cleaning(katara_correction)[:3]
            pickle.dump(katara_correction.keys(), open(os.path.join(b_folder_path, "katara_output.list"), "wb"))
            print "KATARA on", d.name
            print "Precision = {:.2f}".format(er[0])
            print "Recall = {:.2f}".format(er[1])
            print "F1 = {:.2f}".format(er[2])
            print "--------------------"
        if "Min-k" in self.BASELINES:
            fv_folder_path = os.path.join(self.RESULTS_FOLDER, d.name, "feature-vectors")
            fv = {j: {(i, j): [] for i in range(d.dataframe.shape[0])} for j in range(d.dataframe.shape[1])}
            for j, attribute in enumerate(d.dataframe.columns.tolist()):
                o_fv, p_fv, r_fv, k_fv = pickle.load(gzip.open(os.path.join(fv_folder_path, attribute + ".dictionary"), "rb"))
                for i in range(d.dataframe.shape[0]):
                    if "dboost" in self.ERROR_DETECTION_TOOLS:
                        fv[j][(i, j)] += o_fv[(i, j)]
                    if "regex" in self.ERROR_DETECTION_TOOLS:
                        fv[j][(i, j)] += p_fv[(i, j)]
                    if "fd_checker" in self.ERROR_DETECTION_TOOLS:
                        fv[j][(i, j)] += r_fv[(i, j)]
                    if "katara" in self.ERROR_DETECTION_TOOLS:
                        fv[j][(i, j)] += k_fv[(i, j)]
            thresholds_list = [0.0, 0.2, 0.4, 0.6, 0.8]
            max_performance = []
            best_k = 0.0
            for k in thresholds_list:
                print "Min-k = {:.2f}".format(k)
                correction_dictionary = {}
                for j in range(d.dataframe.shape[1]):
                    for cell in fv[j]:
                        if len(fv[j][cell]) > 0 and float(sum(fv[j][cell])) / len(fv[j][cell]) > k:
                            correction_dictionary[cell] = "JUST A DUMMY VALUE"
                er = d.evaluate_data_cleaning(correction_dictionary)[:3]
                if not max_performance or er[2] > max_performance[2]:
                    max_performance = er
                    best_k = k
            print "Min-k on", d.name
            print "Best k", best_k
            print "Precision = {:.2f}".format(max_performance[0])
            print "Recall = {:.2f}".format(max_performance[1])
            print "F1 = {:.2f}".format(max_performance[2])
            print "--------------------"
        if "Maximum Entropy" in self.BASELINES:
            random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
            labeled_tuples = {i: 1 for i in random_tuples_list[:10]}
            correction_dictionary = {}
            while 1:
                best_strategy = ""
                max_p = -1.0
                for strategy_name in list(strategies_output):
                    temp_cd = {}
                    for cell in strategies_output[strategy_name]:
                        temp_cd[cell] = "JUST A DUMMY VALUE"
                    temp_er = d.evaluate_data_cleaning(temp_cd, sampled_rows_dictionary=labeled_tuples)[:3]
                    if temp_er[0] > max_p:
                        max_p = temp_er[0]
                        best_strategy = strategy_name
                for cell in strategies_output[best_strategy]:
                    correction_dictionary[cell] = "JUST A DUMMY VALUE"
                er = d.evaluate_data_cleaning(correction_dictionary)[:3]
                print "Maximum Entropy on", d.name
                print "Labeled Tuples Count = {}".format(len(labeled_tuples))
                print "Precision = {:.2f}".format(er[0])
                print "Recall = {:.2f}".format(er[1])
                print "F1 = {:.2f}".format(er[2])
                print "--------------------"
                if len(labeled_tuples) > 100:
                    break
                for cell in strategies_output[best_strategy]:
                    labeled_tuples[cell[0]] = 1
                strategies_output.pop(best_strategy)
        if "Metadata Driven" in self.BASELINES:
            dboost_output = {cell: 1 for cell in pickle.load(open(os.path.join(b_folder_path, "dboost_output.list"), "rb"))}
            nadeef_output = {cell: 1 for cell in pickle.load(open(os.path.join(b_folder_path, "nadeef_output.list"), "rb"))}
            katara_output = {cell: 1 for cell in pickle.load(open(os.path.join(b_folder_path, "katara_output.list"), "rb"))}
            lfv = {}
            columns_frequent_values = {}
            for j, attribute in enumerate(d.dataframe.columns.tolist()):
                fd = {}
                for value in d.dataframe[attribute].tolist():
                    if value not in fd:
                        fd[value] = 0
                    fd[value] += 1
                sorted_fd = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)[:int(d.dataframe.shape[0] / 10.0)]
                columns_frequent_values[j] = {v: f for v, f in sorted_fd}
            cells_list = list(itertools.product(range(d.dataframe.shape[0]), range(d.dataframe.shape[1])))
            for cell in cells_list:
                lfv[cell] = []
                lfv[cell] += [1 if cell in dboost_output else 0]
                lfv[cell] += [1 if cell in nadeef_output else 0]
                lfv[cell] += [1 if cell in katara_output else 0]
                # -----
                value = d.dataframe.iloc[cell[0], cell[1]]
                lfv[cell] += [1 if value in columns_frequent_values[cell[1]] else 0]
                lfv[cell] += [1 if re.findall(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value) else 0]
                lfv[cell] += [1 if re.findall("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", value) else 0]
                lfv[cell] += [1 if re.findall("^[\d]+$", value) else 0]
                lfv[cell] += [1 if re.findall(r"[\w.-]+@[\w.-]+", value) else 0]
                lfv[cell] += [1 if re.findall("^[\d]{16}$", value) else 0]
                lfv[cell] += [1 if value.lower() in ["m", "f"] else 0]
                lfv[cell] += [1 if re.findall("^[\d]{4,6}$", value) else 0]
                lfv[cell] += [1 if not value else 0]
                for la, ra in dataset_constraints[d.name]["functions"]:
                    lfv[cell] += [1 if d.dataframe.columns.tolist()[cell[1]] in [la, ra] else 0]
            sampling_range = range(10, 101, 10)
            mean_results = {s: [0.0, 0.0, 0.0] for s in sampling_range}
            for r in range(self.RUN_COUNT):
                print "Run {}".format(r)
                random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
                for s in sampling_range:
                    labeled_tuples = {i: 1 for i in random_tuples_list[:s]}
                    x_train = []
                    y_train = []
                    for cell in cells_list:
                        if cell[0] in labeled_tuples:
                            x_train.append(lfv[cell])
                            y_train.append(int(cell in d.actual_errors_dictionary))
                    if sum(y_train) == 0:
                        continue
                    x_test = [lfv[cell] for cell in cells_list]
                    test_cells = [cell for cell in cells_list]
                    if sum(y_train) != len(y_train):
                        model = sklearn.ensemble.AdaBoostClassifier(n_estimators=6)
                        model.fit(x_train, y_train)
                        predicted_labels = model.predict(x_test)
                    else:
                        predicted_labels = len(test_cells) * [1]
                    correction_dictionary = {}
                    for index, pl in enumerate(predicted_labels):
                        cell = test_cells[index]
                        if cell[0] in labeled_tuples:
                            if cell in d.actual_errors_dictionary:
                                correction_dictionary[cell] = "JUST A DUMMY VALUE"
                        elif pl:
                            correction_dictionary[cell] = "JUST A DUMMY VALUE"
                    er = d.evaluate_data_cleaning(correction_dictionary)[:3]
                    mean_results[s] = (numpy.array(mean_results[s]) + numpy.array(er) / self.RUN_COUNT).tolist()
            results_string = "\\addplot coordinates{(0,0.0)"
            for s in sampling_range:
                print "Metadata Driven on", d.name
                print "Labeled Tuples Count = {}".format(s)
                print "Precision = {:.2f}".format(mean_results[s][0])
                print "Recall = {:.2f}".format(mean_results[s][1])
                print "F1 = {:.2f}".format(mean_results[s][2])
                print "--------------------"
                results_string += "({},{:.2f})".format(s, mean_results[s][2])
            results_string += "}; \\addlegendentry{Metadata Driven}"
            print results_string
        if "ActiveClean" in self.BASELINES:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
            text = [" ".join(row) for row in d.dataframe.get_values().tolist()]
            acfv = vectorizer.fit_transform(text).toarray()
            sampling_range = range(1, self.LABELING_BUDGET + 1, 1)
            mean_results = {s: [0.0, 0.0, 0.0] for s in sampling_range}
            # actual_dirty_tuples = {i: 1 for i in range(d.dataframe.shape[0]) if
            #                        int(sum([(i, j) in d.actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0)}
            for r in range(self.RUN_COUNT):
                print "Run {}".format(r)
                labeled_tuples = {}
                adaptive_detector_output = []
                while len(labeled_tuples) < self.LABELING_BUDGET:
                    if len(adaptive_detector_output) < 1:
                        adaptive_detector_output = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
                    labeled_tuples.update({i: 1 for i in numpy.random.choice(adaptive_detector_output, 1, replace=False)})
                    x_train = []
                    y_train = []
                    for i in labeled_tuples:
                        x_train.append(acfv[i, :])
                        y_train.append(int(sum([(i, j) in d.actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0))
                    adaptive_detector_output = []
                    x_test = [acfv[i, :] for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
                    test_rows = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
                    if sum(y_train) == len(y_train):
                        predicted_labels = len(test_rows) * [1]
                    elif sum(y_train) == 0 or len(x_train[0]) == 0:
                        predicted_labels = len(test_rows) * [0]
                    else:
                        model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True)
                        model.fit(x_train, y_train)
                        predicted_labels = model.predict(x_test)
                    correction_dictionary = {}
                    for index, pl in enumerate(predicted_labels):
                        i = test_rows[index]
                        if pl:
                            adaptive_detector_output.append(i)
                            for j in range(d.dataframe.shape[1]):
                                correction_dictionary[(i, j)] = "JUST A DUMMY VALUE"
                    for i in labeled_tuples:
                        for j in range(d.dataframe.shape[1]):
                            correction_dictionary[(i, j)] = "JUST A DUMMY VALUE"
                    er = d.evaluate_data_cleaning(correction_dictionary)[:3]
                    mean_results[len(labeled_tuples)] = (numpy.array(mean_results[len(labeled_tuples)]) + numpy.array(er) / self.RUN_COUNT).tolist()
                    # # Tuple-wise evaluation
                    # tp = 0.0
                    # outputted_tuples = {}
                    # for i, j in correction_dictionary:
                    #     if i not in outputted_tuples:
                    #         outputted_tuples[i] = 1
                    #         if i in actual_dirty_tuples:
                    #             tp += 1.0
                    # p = tp / len(outputted_tuples)
                    # r = tp / len(actual_dirty_tuples)
                    # f = 0.0 if (p + r) == 0.0 else (2 * p * r) / (p + r)
                    # mean_results[len(labeled_tuples)] = (numpy.array(mean_results[len(labeled_tuples)]) + numpy.array([p, r, f]) / self.RUN_COUNT).tolist()
            results_string = "\\addplot coordinates{(0,0.0)"
            for s in sampling_range:
                print "ActiveClean on", d.name
                print "Labeled Tuples Count = {}".format(s)
                print "Precision = {:.2f}".format(mean_results[s][0])
                print "Recall = {:.2f}".format(mean_results[s][1])
                print "F1 = {:.2f}".format(mean_results[s][2])
                print "--------------------"
                results_string += "({},{:.2f})".format(s, mean_results[s][2])
            results_string += "}; \\addlegendentry{ActiveClean}"
            print results_string
########################################


########################################
if __name__ == "__main__":
    # --------------------
    application = Raha()
    # --------------------

    print "===================== Dataset: {} =====================".format(myGlobals.d.name)
    # --------------------
    application.strategy_profiler(myGlobals.d)
    application.dataset_profiler(myGlobals.d)
    application.evaluation_profiler(myGlobals.d)
    # --------------------
    application.feature_generator(myGlobals.d)
    #application.error_detector(myGlobals.d)
    # --------------------
    #application.baselines(myGlobals.d)
    # --------------------
########################################
