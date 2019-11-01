########################################
# Raha
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import sys
import re
import math
import time
import string
import json
import random
import pickle
import tempfile
import itertools
import hashlib

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
import multiprocessing
# import IPython.display
# import ipywidgets

try:
    from . import dataset
    from .tools.katara import katara
    from .tools.dBoost.dboost import imported_dboost
except:
    import dataset
    from tools.katara import katara
    from tools.dBoost.dboost import imported_dboost
########################################


########################################
class Raha:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.RUN_COUNT = 1
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0   # [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        self.ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]   # ["OD", "PVD", "RVD", "KBVD", "TFIDF"]
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = "homogeneity"   # ["homogeneity", "majority"]
        self.CLUSTERING_BASED_SAMPLING = True
        self.SAVE_STRATEGY_OUTPUT = True

    def strategy_runner_process(self, args):
        """
        This method runs an error detection strategy in a parallel process.
        """
        d, algorithm, configuration = args
        start_time = time.time()
        strategy_name = json.dumps([algorithm, configuration])
        outputted_cells = {}
        if algorithm == "OD":
            random_string = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
            dataset_path = os.path.join(tempfile.gettempdir(), d.name + "-" + random_string + ".csv")
            d.write_csv_dataset(dataset_path, d.dataframe)
            configuration[0] = "--" + configuration[0]
            params = ["-F", ",", "--statistical", "0.5"] + configuration + [dataset_path]
            imported_dboost.run_dboost(params)
            algorithm_results_path = dataset_path + "-dboost_output.csv"
            if os.path.exists(algorithm_results_path):
                ocdf = pandas.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
                os.remove(algorithm_results_path)
            os.remove(dataset_path)
        elif algorithm == "PVD":
            attribute, ch = configuration
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].iteritems():
                try:
                    if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                        outputted_cells[(i, j)] = ""
                except:
                    continue
        elif algorithm == "RVD":
            l_attribute, r_attribute = configuration
            l_j = d.dataframe.columns.get_loc(l_attribute)
            r_j = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            for i, row in d.dataframe.iterrows():
                if row[l_attribute]:
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    if row[r_attribute]:
                        value_dictionary[row[l_attribute]][row[r_attribute]] = 1
            for i, row in d.dataframe.iterrows():
                if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
                    outputted_cells[(i, l_j)] = ""
                    outputted_cells[(i, r_j)] = ""
        elif algorithm == "KBVD":
            outputted_cells = katara.run_katara(d, configuration)
        detected_cells_list = list(outputted_cells.keys())
        strategy_profile = {
            "name": strategy_name,
            "output": detected_cells_list,
            "runtime": time.time() - start_time
        }
        if self.SAVE_STRATEGY_OUTPUT:
            sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
            file_name = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16)) + ".dictionary"
            pickle.dump(strategy_profile, open(os.path.join(sp_folder_path, file_name), "wb"))
        print("{} cells are detected by {}.".format(len(detected_cells_list), strategy_name))
        return strategy_profile

    def feature_extractor_process(self, args):
        """
        This method extracts features for a given data column in a parallel process.
        """
        d, j, strategy_profiles = args
        feature_vectors = numpy.zeros((d.dataframe.shape[0], len(strategy_profiles)))
        for strategy_index, strategy_profile in enumerate(strategy_profiles):
            strategy_name = json.loads(strategy_profile["name"])[0]
            if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                for cell in strategy_profile["output"]:
                    if cell[1] == j:
                        feature_vectors[cell[0], strategy_index] = 1.0
        if "TFIDF" in self.ERROR_DETECTION_ALGORITHMS:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, max_df=0, stop_words="english")
            corpus = d.dataframe.iloc[:, j]
            try:
                tfidf_features = vectorizer.fit_transform(corpus)
                feature_vectors = numpy.column_stack((feature_vectors, tfidf_features.todense()))
            except:
                pass
        non_identical_columns = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
        feature_vectors = feature_vectors[:, non_identical_columns]
        print("{} Features Extracted for column {}.".format(feature_vectors.shape[1], j))
        return feature_vectors

    def cluster_builder_process(self, args):
        """
        This method builds a clustering model for a given data column in a parallel process.
        """
        d, j, feature_vectors = args
        clustering_range = range(2, self.LABELING_BUDGET + 2)
        clusters_j_k_c_ce = {k: {} for k in clustering_range}
        cells_clusters_j_k_ce = {k: {} for k in clustering_range}
        if feature_vectors.any():
            clustering_model = scipy.cluster.hierarchy.linkage(feature_vectors, method="average", metric="cosine")
            for k in clusters_j_k_c_ce:
                model_labels = [l - 1 for l in scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                for index, c in enumerate(model_labels):
                    if c not in clusters_j_k_c_ce[k]:
                        clusters_j_k_c_ce[k][c] = {}
                    cell = (index, j)
                    clusters_j_k_c_ce[k][c][cell] = feature_vectors[index]
                    cells_clusters_j_k_ce[k][cell] = c
        print("A clustering model built for column {}.".format(j))
        return [clusters_j_k_c_ce, cells_clusters_j_k_ce]

    def classification_process(self, args):
        """
        This method trains and tests a classification model for a given data column in a parallel process.
        """
        d, j, feature_vectors, labeled_tuples, extended_labeled_cells = args
        x_train = [feature_vectors[i, :] for i in range(d.dataframe.shape[0]) if (i, j) in extended_labeled_cells]
        y_train = [extended_labeled_cells[(i, j)] for i in range(d.dataframe.shape[0]) if (i, j) in extended_labeled_cells]
        x_test = feature_vectors
        if sum(y_train) == len(y_train):
            predicted_labels = numpy.ones(d.dataframe.shape[0])
        elif sum(y_train) == 0 or len(x_train[0]) == 0:
            predicted_labels = numpy.zeros(d.dataframe.shape[0])
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
        correction_dictionary = {}
        for i, pl in enumerate(predicted_labels):
            if (i in labeled_tuples and extended_labeled_cells[(i, j)]) or (i not in labeled_tuples and pl):
                correction_dictionary[(i, j)] = "JUST A DUMMY VALUE"
        return correction_dictionary

    def run(self, dd):
        """
        This method runs Raha on an input dataset.
        """
        # --------------------Initializing the Dataset Instance--------------------
        d = dataset.Dataset(dd)
        d.results_folder = os.path.join(os.path.dirname(dd["path"]), "raha-results-" + d.name)
        if not os.path.exists(d.results_folder):
            os.mkdir(d.results_folder)
        # --------------------Running Error Detection Strategies--------------------
        sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
        if os.path.exists(sp_folder_path):
            sys.stderr.write("Since the error detection strategies have already been run on the dataset, I just load their results!\n")
            strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                      for strategy_file in os.listdir(sp_folder_path)]
        else:
            if self.SAVE_STRATEGY_OUTPUT:
                os.mkdir(sp_folder_path)
            algorithm_and_configurations = []
            for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
                if algorithm_name == "OD":
                    configuration_list = [
                        list(a) for a in
                        list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                               ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                        list(itertools.product(["gaussian"], ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
                    algorithm_and_configurations.extend([[d, algorithm_name, configuration] for configuration in configuration_list])
                elif algorithm_name == "PVD":
                    configuration_list = []
                    for attribute in d.dataframe.columns:
                        column_data = "".join(d.dataframe[attribute].tolist())
                        characters_dictionary = {ch: 1 for ch in column_data}
                        for ch in characters_dictionary:
                            configuration_list.append([attribute, ch])
                    algorithm_and_configurations.extend([[d, algorithm_name, configuration] for configuration in configuration_list])
                elif algorithm_name == "RVD":
                    al = d.dataframe.columns.tolist()
                    configuration_list = [[a, b] for (a, b) in itertools.product(al, al) if a != b]
                    algorithm_and_configurations.extend([[d, algorithm_name, configuration] for configuration in configuration_list])
                elif algorithm_name == "KBVD":
                    configuration_list = [os.path.join(os.path.dirname(__file__), "tools", "katara", "knowledge-base", p)
                                          for p in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "katara", "knowledge-base"))]
                    algorithm_and_configurations.extend([[d, algorithm_name, configuration] for configuration in configuration_list])
            random.shuffle(algorithm_and_configurations)
            pool = multiprocessing.Pool()
            strategy_profiles_list = pool.map(self.strategy_runner_process, algorithm_and_configurations)
        # --------------------Generating Features--------------------
        fe_args = [[d, j, strategy_profiles_list] for j in range(d.dataframe.shape[1])]
        pool = multiprocessing.Pool()
        columns_features_list = pool.map(self.feature_extractor_process, fe_args)
        # --------------------Building the Hierarchical Clustering Model--------------------
        bc_args = [[d, j, feature_vectors] for j, feature_vectors in enumerate(columns_features_list)]
        pool = multiprocessing.Pool()
        clustering_results = pool.map(self.cluster_builder_process, bc_args)
        clusters_j_k_c_ce = {}
        cells_clusters_j_k_ce = {}
        for j, result in enumerate(clustering_results):
            clusters_j_k_c_ce[j] = clustering_results[j][0]
            cells_clusters_j_k_ce[j] = clustering_results[j][1]
        clustering_range = range(2, self.LABELING_BUDGET + 2)
        clusters_k_j_c_ce = {k: {j: clusters_j_k_c_ce[j][k] for j in range(d.dataframe.shape[1])} for k in clustering_range}
        cells_clusters_k_j_ce = {k: {j: cells_clusters_j_k_ce[j][k] for j in range(d.dataframe.shape[1])} for k in clustering_range}
        # --------------------Iterative Labeling and Learning--------------------
        ed_folder_path = os.path.join(d.results_folder, "error-detection")
        if not os.path.exists(ed_folder_path):
            os.mkdir(ed_folder_path)
        labeled_tuples = {}
        labeled_cells = {}
        extended_labeled_cells = {}
        correction_dictionary = {}
        for k in clusters_k_j_c_ce:
            # --------------------Calculating Number of Labels per Clusters--------------------
            labels_per_cluster = {}
            for j in range(d.dataframe.shape[1]):
                for c in clusters_k_j_c_ce[k][j]:
                    labels_per_cluster[(j, c)] = {cell: labeled_cells[cell] for cell in clusters_k_j_c_ce[k][j][c] if cell[0] in labeled_tuples}
            # --------------------Sampling a Tuple--------------------
            if self.CLUSTERING_BASED_SAMPLING:
                tuple_score = numpy.zeros(d.dataframe.shape[0])
                for i in range(d.dataframe.shape[0]):
                    if i not in labeled_tuples:
                        score = 0.0
                        for j in range(d.dataframe.shape[1]):
                            if clusters_k_j_c_ce[k][j]:
                                cell = (i, j)
                                c = cells_clusters_k_j_ce[k][j][cell]
                                score += math.exp(-len(labels_per_cluster[(j, c)]))
                        tuple_score[i] = math.exp(score)
            else:
                tuple_score = numpy.ones(d.dataframe.shape[0])
            sum_tuple_score = sum(tuple_score)
            p_tuple_score = tuple_score / sum_tuple_score
            si = numpy.random.choice(numpy.arange(d.dataframe.shape[0]), 1, p=p_tuple_score)[0]
            # si, score = max(tuple_score.iteritems(), key=operator.itemgetter(1))
            # --------------------Labeling the Tuple--------------------
            labeled_tuples[si] = tuple_score[si]
            if hasattr(d, "clean_dataframe"):
                actual_errors_dictionary = d.get_actual_errors_dictionary()
                for j in range(d.dataframe.shape[1]):
                    cell = (si, j)
                    user_label = int(cell in actual_errors_dictionary)
                    if random.random() > self.USER_LABELING_ACCURACY:
                        user_label = 1 - user_label
                    labeled_cells[cell] = user_label
                    if cell in cells_clusters_k_j_ce[k][j]:
                        c = cells_clusters_k_j_ce[k][j][cell]
                        labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
            else:
                print("Label the dirty cells in the following sampled tuple.")
                sampled_tuple = pandas.DataFrame(data=[d.dataframe.iloc[si, :]], columns=d.dataframe.columns)
                # IPython.display.display(sampled_tuple)
                for j in range(d.dataframe.shape[1]):
                    cell = (si, j)
                    value = d.dataframe.iloc[cell]
                    labeled_cells[cell] = int(input("Is the value '{}' dirty?\nType 1 for yes.\nType 0 for no.\n".format(value)))
                    if cell in cells_clusters_k_j_ce[k][j]:
                        c = cells_clusters_k_j_ce[k][j][cell]
                        labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
            # --------------------Propagating User Labels Through the Clusters--------------------
            extended_labeled_cells = dict(labeled_cells)
            if self.CLUSTERING_BASED_SAMPLING:
                for j in clusters_k_j_c_ce[k]:
                    for c in clusters_k_j_c_ce[k][j]:
                        if len(labels_per_cluster[(j, c)]) > 0:
                            if self.LABEL_PROPAGATION_METHOD == "homogeneity":
                                cluster_label = list(labels_per_cluster[(j, c)].values())[0]
                                if sum(labels_per_cluster[(j, c)].values()) in [0, len(labels_per_cluster[(j, c)])]:
                                    for cell in clusters_k_j_c_ce[k][j][c]:
                                        extended_labeled_cells[cell] = cluster_label
                            elif self.LABEL_PROPAGATION_METHOD == "majority":
                                cluster_label = round(sum(labels_per_cluster[(j, c)].values()) / len(labels_per_cluster[(j, c)]))
                                for cell in clusters_k_j_c_ce[k][j][c]:
                                    extended_labeled_cells[cell] = cluster_label
        # --------------------Training and Testing Classification Models--------------------
        c_args = [[d, j, columns_features_list[j], labeled_tuples, extended_labeled_cells] for j in range(d.dataframe.shape[1])]
        pool = multiprocessing.Pool()
        columns_correction_dictionaries_list = pool.map(self.classification_process, c_args)
        for column_correction_dictionary in columns_correction_dictionaries_list:
            correction_dictionary.update(column_correction_dictionary)
        # IPython.display.display(d.dataframe.style.apply(
        #    lambda x: ["background-color: red" if (i, d.dataframe.columns.get_loc(x.name)) in correction_dictionary else ""
        #              for i, cv in enumerate(x)]))
        # if not hasattr(d, "clean_dataframe"):
        #     continue_flag = int(input("Would you like to label one more tuple?\nType 1 for yes.\nType 0 for no.\n"))
        #     if not continue_flag:
        #         break
        pickle.dump(correction_dictionary, open(os.path.join(ed_folder_path, "correction.dictionary"), "wb"))
        return correction_dictionary

    def benchmark(self, dd):
        """
        This method benchmarks Raha.
        """
        d = dataset.Dataset(dd)
        sampling_range = [self.LABELING_BUDGET]
        aggregate_results = {s: [] for s in sampling_range}
        for r in range(self.RUN_COUNT):
            print("Run {}...".format(r))
            for s in sampling_range:
                self.LABELING_BUDGET = s
                correction_dictionary = self.run(dd)
                er = d.get_data_cleaning_evaluation(correction_dictionary)[:3]
                aggregate_results[s].append(er)
        results_string = "\\addplot[error bars/.cd,y dir=both,y explicit] coordinates{(0,0.0)"
        for s in sampling_range:
            mean = numpy.mean(numpy.array(aggregate_results[s]), axis=0)
            std = numpy.std(numpy.array(aggregate_results[s]), axis=0)
            print("Raha on {}".format(d.name))
            print("Labeled Tuples Count = {}".format(s))
            print("Precision = {:.2f} +- {:.2f}".format(mean[0], std[0]))
            print("Recall = {:.2f} +- {:.2f}".format(mean[1], std[1]))
            print("F1 = {:.2f} +- {:.2f}".format(mean[2], std[2]))
            print("--------------------")
            results_string += "({},{:.2f})+-(0,{:.2f})".format(s, mean[2], std[2])
        results_string += "}; \\addlegendentry{Raha}"
        print(results_string)
########################################


########################################
if __name__ == "__main__":
    dataset_name = "hospital"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = Raha()
    # app.run(dataset_dictionary)
    app.benchmark(dataset_dictionary)
########################################
