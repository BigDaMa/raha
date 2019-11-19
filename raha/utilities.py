########################################
# Utilities
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
import math
import json
import pickle
import random
import operator
import itertools

import numpy
import sklearn
import scipy.cluster
import scipy.spatial

import raha.dataset
########################################


########################################
def get_tuple_wise_evaluation(d, correction_dictionary):
    """
    This method evaluates data cleaning in tuple-wise manner.
    """
    actual_errors_dictionary = d.get_actual_errors_dictionary()
    actual_dirty_tuples = {i: 1 for i in range(d.dataframe.shape[0]) if int(sum([(i, j) in actual_errors_dictionary
                           for j in range(d.dataframe.shape[1])]) > 0)}
    tp = 0.0
    outputted_tuples = {}
    for i, j in correction_dictionary:
        if i not in outputted_tuples:
            outputted_tuples[i] = 1
            if i in actual_dirty_tuples:
                tp += 1.0
    p = tp / len(outputted_tuples)
    r = tp / len(actual_dirty_tuples)
    f = 0.0 if (p + r) == 0.0 else (2 * p * r) / (p + r)
    return p, r, f


def dataset_profiler(dataset_dictionary):
    """
    This method profiles the columns of dataset.
    """
    print("------------------------------------------------------------------------\n"
          "--------------------------Profiling the Dataset-------------------------\n"
          "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    dp_folder_path = os.path.join(d.results_folder, "dataset-profiling")
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
            if value not in values_dictionary:
                values_dictionary[value] = 0.0
            values_dictionary[value] += 1.0
        column_profile = {
            "characters": {ch: characters_dictionary[ch] / d.dataframe.shape[0] for ch in characters_dictionary},
            "values": {v: values_dictionary[v] / d.dataframe.shape[0] for v in values_dictionary},
        }
        pickle.dump(column_profile, open(os.path.join(dp_folder_path, attribute + ".dictionary"), "wb"))


def evaluation_profiler(dataset_dictionary):
    """
    This method computes the performance of the error detection strategies on historical data.
    """
    print("------------------------------------------------------------------------\n"
          "---------Profiling the Performance of Strategies on the Dataset---------\n"
          "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    actual_errors_dictionary = d.get_actual_errors_dictionary()
    ep_folder_path = os.path.join(d.results_folder, "evaluation-profiling")
    if not os.path.exists(ep_folder_path):
        os.mkdir(ep_folder_path)
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    columns_performance = {j: {} for j in range(d.dataframe.shape[1])}
    strategies_file_list = os.listdir(sp_folder_path)
    for strategy_file in strategies_file_list:
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategy_name = strategy_profile["name"]
        strategy_output = strategy_profile["output"]
        for column_index, attribute in enumerate(d.dataframe.columns.tolist()):
            actual_column_errors = {(i, j): 1 for (i, j) in actual_errors_dictionary if j == column_index}
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
    for j, attribute in enumerate(d.dataframe.columns.tolist()):
        pickle.dump(columns_performance[j], open(os.path.join(ep_folder_path, attribute + ".dictionary"), "wb"))


def get_selected_strategies_via_historical_data(dataset_dictionary, historical_dataset_dictionaries):
    """
    This method uses historical data to rank error detection strategies for the dataset and select the top-ranked.
    """
    print("------------------------------------------------------------------------\n"
          "-------Selecting Promising Strategies Based on Historical Datasets------\n"
          "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    columns_similarity = {}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        ndp_folder_path = os.path.join(d.results_folder, "dataset-profiling")
        ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    hdp_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-results-" + hdd["name"])
                    hcp = pickle.load(open(os.path.join(hdp_folder_path, "dataset-profiling", ha + ".dictionary"), "rb"))
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
    f1_measure = {}
    for hdd in historical_dataset_dictionaries:
        if hdd["name"] != d.name:
            hd = raha.dataset.Dataset(hdd)
            for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                ep_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-results-" + hdd["name"], "evaluation-profiling")
                strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                if (hd.name, ha) not in f1_measure:
                    f1_measure[(hd.name, ha)] = {}
                for strategy_name in strategies_performance:
                    f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
    strategies_score = {a: {} for a in d.dataframe.columns.tolist()}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    similarity = columns_similarity[(d.name, na, hd.name, ha)]
                    if similarity == 0:
                        continue
                    for strategy_name in f1_measure[(hd.name, ha)]:
                        score = similarity * f1_measure[(hd.name, ha)][strategy_name]
                        if score <= 0.0:
                            continue
                        sn = json.loads(strategy_name)
                        if sn[0] == "OD" or sn[0] == "KBVD":
                            if strategy_name not in strategies_score[na] or score >= strategies_score[na][strategy_name]:
                                strategies_score[na][strategy_name] = score
                        elif sn[0] == "PVD":
                            sn[1][0] = na
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                        elif sn[0] == "RVD":
                            this_a_i = sn[1].index(ha)
                            that_a = sn[1][1 - this_a_i]
                            most_similar_a = d.dataframe.columns.tolist()[0]
                            most_similar_v = -1
                            for aa in d.dataframe.columns.tolist():
                                if aa != na and columns_similarity[(d.name, aa, hd.name, that_a)] > most_similar_v:
                                    most_similar_v = columns_similarity[(d.name, aa, hd.name, that_a)]
                                    most_similar_a = aa
                            sn[1][this_a_i] = na
                            sn[1][1 - this_a_i] = most_similar_a
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                        else:
                            sys.stderr.write("I do not know this error detection tool!\n")
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    strategies_output = {}
    strategies_runtime = {}
    selected_strategy_profiles = []
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategies_output[strategy_profile["name"]] = strategy_profile["output"]
        strategies_runtime[strategy_profile["name"]] = strategy_profile["runtime"]
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
            if snd[0] == "OD" or snd[0] == "KBVD":
                runtime = strategies_runtime[sn] / d.dataframe.shape[1]
            elif snd[0] == "PVD":
                runtime = strategies_runtime[sn]
            elif snd[0] == "RVD":
                runtime = strategies_runtime[sn] / 2
            else:
                sys.stderr.write("I do not know this error detection tool!\n")
            strategy_profile = {
                "name": sn,
                "output": [cell for cell in strategies_output[sn] if d.dataframe.columns.tolist()[cell[1]] == a],
                "runtime": runtime
            }
            selected_strategy_profiles.append(strategy_profile)
    return selected_strategy_profiles


def get_selected_strategies_via_ground_truth(dataset_dictionary, strategies_count):
    """
    This method uses the ground truth to rank error detection strategies for the dataset.
    """
    print("------------------------------------------------------------------------\n"
          "---Selecting Worst, Random, and Best Strategies Based on Ground Truth---\n"
          "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    f1_measure = {}
    ep_folder_path = os.path.join(d.results_folder, "evaluation-profiling")
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        strategies_performance = pickle.load(open(os.path.join(ep_folder_path, na + ".dictionary"), "rb"))
        for strategy_name in strategies_performance:
            f1_measure[(na, strategy_name)] = strategies_performance[strategy_name][2]
    sorted_f1_measure = sorted(f1_measure.items(), key=operator.itemgetter(1))
    worst_strategies = {s: f1 for s, f1 in sorted_f1_measure[:strategies_count]}
    random_strategies = {s: f1 for s, f1 in [sorted_f1_measure[i] for i in
                                             random.sample(range(len(sorted_f1_measure)), strategies_count)]}
    best_strategies = {s: f1 for s, f1 in sorted_f1_measure[-strategies_count:]}
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    worst_strategy_profiles = []
    random_strategy_profiles = []
    best_strategy_profiles = []
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        for a in d.dataframe.columns.tolist():
            snd = json.loads(strategy_profile["name"])
            runtime = 0.0
            if snd[0] == "OD" or snd[0] == "KBVD":
                runtime = strategy_profile["runtime"] / d.dataframe.shape[1]
            elif snd[0] == "PVD":
                runtime = strategy_profile["runtime"]
            elif snd[0] == "RVD":
                runtime = strategy_profile["runtime"] / 2
            else:
                sys.stderr.write("I do not know this error detection tool!\n")
            sp = {
                "name": strategy_profile["name"],
                "output": [cell for cell in strategy_profile["output"] if d.dataframe.columns.tolist()[cell[1]] == a],
                "runtime": runtime
            }
            if (a, strategy_profile["name"]) in worst_strategies:
                worst_strategy_profiles.append(sp)
            if (a, strategy_profile["name"]) in random_strategies:
                random_strategy_profiles.append(sp)
            if (a, strategy_profile["name"]) in best_strategies:
                best_strategy_profiles.append(sp)
    return worst_strategy_profiles, random_strategy_profiles, best_strategy_profiles


def get_strategies_count_and_runtime(dataset_dictionary):
    """
    This method calculates the number of all strategies and their total runtime.
    """
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    strategies_count = 0
    strategies_runtime = 0
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategies_runtime += strategy_profile["runtime"]
        sn = json.loads(strategy_profile["name"])
        if sn[0] in ["OD", "KBVD"]:
            strategies_count += d.dataframe.shape[1]
        if sn[0] in ["PVD", "RVD"]:
            strategies_count += 1
    return strategies_count, strategies_runtime


def error_detection_with_selected_strategies(dataset_dictionary, strategy_profiles_list):
    """
    This method runs Raha on an input dataset to detection data errors with only the given strategy profiles.
    """
    def feature_extractor_process(args):
        """
        This method extracts features for a given data column in a parallel process.
        """
        d, j, strategy_profiles = args
        feature_vectors = numpy.zeros((d.dataframe.shape[0], len(strategy_profiles)))
        for strategy_index, strategy_profile in enumerate(strategy_profiles):
            strategy_name = json.loads(strategy_profile["name"])[0]
            if strategy_name in ERROR_DETECTION_ALGORITHMS:
                for cell in strategy_profile["output"]:
                    if cell[1] == j:
                        feature_vectors[cell[0], strategy_index] = 1.0
        if "TFIDF" in ERROR_DETECTION_ALGORITHMS:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
            corpus = d.dataframe.iloc[:, j]
            try:
                tfidf_features = vectorizer.fit_transform(corpus)
                feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
            except:
                pass
        non_identical_columns = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
        feature_vectors = feature_vectors[:, non_identical_columns]
        return feature_vectors

    def cluster_builder_process(args):
        """
        This method builds a hierarchical clustering model for a given data column in a parallel process.
        """
        d, j, feature_vectors = args
        clustering_range = range(2, LABELING_BUDGET + 2)
        clusters_k_c_ce = {k: {} for k in clustering_range}
        cells_clusters_k_ce = {k: {} for k in clustering_range}
        try:
            clustering_model = scipy.cluster.hierarchy.linkage(feature_vectors, method="average", metric="cosine")
            for k in clusters_k_c_ce:
                model_labels = [l - 1 for l in scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                for index, c in enumerate(model_labels):
                    if c not in clusters_k_c_ce[k]:
                        clusters_k_c_ce[k][c] = {}
                    cell = (index, j)
                    clusters_k_c_ce[k][c][cell] = 1
                    cells_clusters_k_ce[k][cell] = c
        except:
            pass
        return [clusters_k_c_ce, cells_clusters_k_ce]

    def classification_process(args):
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
            if CLASSIFICATION_MODEL == "ABC":
                classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
            if CLASSIFICATION_MODEL == "DTC":
                classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
            if CLASSIFICATION_MODEL == "GBC":
                classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
            if CLASSIFICATION_MODEL == "GNB":
                classification_model = sklearn.naive_bayes.GaussianNB()
            if CLASSIFICATION_MODEL == "KNC":
                classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if CLASSIFICATION_MODEL == "SGDC":
                classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
            if CLASSIFICATION_MODEL == "SVC":
                classification_model = sklearn.svm.SVC(kernel="sigmoid")
            classification_model.fit(x_train, y_train)
            predicted_labels = classification_model.predict(x_test)
        detection_dictionary = {}
        for i, pl in enumerate(predicted_labels):
            if (i in labeled_tuples and extended_labeled_cells[(i, j)]) or (i not in labeled_tuples and pl):
                detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
        return detection_dictionary

    LABELING_BUDGET = 20
    USER_LABELING_ACCURACY = 1.0
    CLUSTERING_BASED_SAMPLING = True
    CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
    LABEL_PROPAGATION_METHOD = "homogeneity"  # ["homogeneity", "majority"]
    ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]  # ["OD", "PVD", "RVD", "KBVD", "TFIDF"]
    print("------------------------------------------------------------------------\n"
          "--------------------Instantiating the Dataset Object--------------------\n"
          "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-results-" + d.name)
    print("------------------------------------------------------------------------\n"
          "-----------------------Generating Feature Vectors-----------------------\n"
          "------------------------------------------------------------------------")
    columns_features_list = []
    for j in range(d.dataframe.shape[1]):
        fe_args = [d, j, strategy_profiles_list]
        columns_features_list.append(feature_extractor_process(fe_args))
    print("------------------------------------------------------------------------\n"
          "---------------Building the Hierarchical Clustering Model---------------\n"
          "------------------------------------------------------------------------")
    clustering_results = []
    for j in range(d.dataframe.shape[1]):
        bc_args = [d, j, columns_features_list[j]]
        clustering_results.append(cluster_builder_process(bc_args))
    clustering_range = range(2, LABELING_BUDGET + 2)
    clusters_k_j_c_ce = {k: {j: clustering_results[j][0][k] for j in range(d.dataframe.shape[1])} for k in clustering_range}
    cells_clusters_k_j_ce = {k: {j: clustering_results[j][1][k] for j in range(d.dataframe.shape[1])} for k in clustering_range}
    print("------------------------------------------------------------------------\n"
          "-------------------Iterative Clustering-Based Labeling------------------\n"
          "------------------------------------------------------------------------")
    labeled_tuples = {}
    labeled_cells = {}
    labels_per_cluster = {}
    for k in clusters_k_j_c_ce:
        # --------------------Calculating Number of Labels per Clusters--------------------
        for j in range(d.dataframe.shape[1]):
            for c in clusters_k_j_c_ce[k][j]:
                labels_per_cluster[(j, c)] = {cell: labeled_cells[cell] for cell in clusters_k_j_c_ce[k][j][c] if cell[0] in labeled_tuples}
        # --------------------Sampling a Tuple--------------------
        if CLUSTERING_BASED_SAMPLING:
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
        # --------------------Labeling the Tuple--------------------
        labeled_tuples[si] = 1
        if hasattr(d, "clean_dataframe"):
            actual_errors_dictionary = d.get_actual_errors_dictionary()
            for j in range(d.dataframe.shape[1]):
                cell = (si, j)
                user_label = int(cell in actual_errors_dictionary)
                if random.random() > USER_LABELING_ACCURACY:
                    user_label = 1 - user_label
                labeled_cells[cell] = user_label
                if cell in cells_clusters_k_j_ce[k][j]:
                    c = cells_clusters_k_j_ce[k][j][cell]
                    labels_per_cluster[(j, c)][cell] = labeled_cells[cell]
    print("------------------------------------------------------------------------\n"
          "--------------Propagating User Labels Through the Clusters--------------\n"
          "------------------------------------------------------------------------")
    extended_labeled_cells = dict(labeled_cells)
    if CLUSTERING_BASED_SAMPLING:
        k = clustering_range[-1]
        for j in clusters_k_j_c_ce[k]:
            for c in clusters_k_j_c_ce[k][j]:
                if len(labels_per_cluster[(j, c)]) > 0:
                    if LABEL_PROPAGATION_METHOD == "homogeneity":
                        cluster_label = list(labels_per_cluster[(j, c)].values())[0]
                        if sum(labels_per_cluster[(j, c)].values()) in [0, len(labels_per_cluster[(j, c)])]:
                            for cell in clusters_k_j_c_ce[k][j][c]:
                                extended_labeled_cells[cell] = cluster_label
                    elif LABEL_PROPAGATION_METHOD == "majority":
                        cluster_label = round(sum(labels_per_cluster[(j, c)].values()) / len(labels_per_cluster[(j, c)]))
                        for cell in clusters_k_j_c_ce[k][j][c]:
                            extended_labeled_cells[cell] = cluster_label
    print("------------------------------------------------------------------------\n"
          "---------------Training and Testing Classification Models---------------\n"
          "------------------------------------------------------------------------")
    detection_dictionary = {}
    for j in range(d.dataframe.shape[1]):
        c_args = [d, j, columns_features_list[j], labeled_tuples, extended_labeled_cells]
        detection_dictionary.update(classification_process(c_args))
    return detection_dictionary

