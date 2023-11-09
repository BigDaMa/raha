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

import scipy.spatial

import raha
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
    p = 0.0 if len(outputted_tuples) == 0 else tp / len(outputted_tuples)
    r = 0.0 if len(actual_dirty_tuples) == 0 else tp / len(actual_dirty_tuples)
    f = 0.0 if (p + r) == 0.0 else (2 * p * r) / (p + r)
    return p, r, f


def dataset_profiler(dataset_dictionary):
    """
    This method profiles the columns of dataset.
    """
    # print("------------------------------------------------------------------------\n"
    #       "--------------------------Profiling the Dataset-------------------------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
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
    # print("------------------------------------------------------------------------\n"
    #       "---------Profiling the Performance of Strategies on the Dataset---------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
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
    # print("------------------------------------------------------------------------\n"
    #       "-------Selecting Promising Strategies Based on Historical Datasets------\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
    columns_similarity = {}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        ndp_folder_path = os.path.join(d.results_folder, "dataset-profiling")
        ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    hdp_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-baran-results-" + hdd["name"])
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
                ep_folder_path = os.path.join(os.path.dirname(hdd["path"]), "raha-baran-results-" + hdd["name"], "evaluation-profiling")
                strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                if (hd.name, ha) not in f1_measure:
                    f1_measure[(hd.name, ha)] = {}
                for strategy_name in strategies_performance:
                    f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
    strategies_score = {a: {} for a in d.dataframe.columns.tolist()}
    strategies_anchor = {a: {} for a in d.dataframe.columns.tolist()}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        for hdd in historical_dataset_dictionaries:
            if hdd["name"] != d.name:
                hd = raha.dataset.Dataset(hdd)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    similarity = columns_similarity[(d.name, na, hd.name, ha)]
                    anchor = [d.name, na, hd.name, ha]
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
                                strategies_anchor[na][strategy_name] = anchor
                        elif sn[0] == "PVD":
                            sn[1][0] = na
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                                strategies_anchor[na][json.dumps(sn)] = anchor
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
                                strategies_anchor[na][json.dumps(sn)] = anchor
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
                "runtime": runtime,
                "score": good_strategies[sn],
                "new_column": strategies_anchor[a][sn][0] + "." + strategies_anchor[a][sn][1],
                "historical_column": strategies_anchor[a][sn][2] + "." + strategies_anchor[a][sn][3]
            }
            selected_strategy_profiles.append(strategy_profile)
    return selected_strategy_profiles


def get_selected_strategies_via_ground_truth(dataset_dictionary, strategies_count):
    """
    This method uses the ground truth to rank error detection strategies for the dataset.
    """
    # print("------------------------------------------------------------------------\n"
    #       "---Selecting Worst, Random, and Best Strategies Based on Ground Truth---\n"
    #       "------------------------------------------------------------------------")
    d = raha.dataset.Dataset(dataset_dictionary)
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
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
                "runtime": runtime,
                "score": f1_measure[(a, strategy_profile["name"])]
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
    d.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + d.name)
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
    app = raha.Detection()
    # print("------------------------------------------------------------------------\n"
    #       "--------------------Instantiating the Dataset Object--------------------\n"
    #       "------------------------------------------------------------------------")
    d = app.initialize_dataset(dataset_dictionary)
    # print("------------------------------------------------------------------------\n"
    #       "-------------------Running Error Detection Strategies-------------------\n"
    #       "------------------------------------------------------------------------")
    d.strategy_profiles = strategy_profiles_list
    # print("------------------------------------------------------------------------\n"
    #       "-----------------------Generating Feature Vectors-----------------------\n"
    #       "------------------------------------------------------------------------")
    app.generate_features(d)
    # print("------------------------------------------------------------------------\n"
    #       "---------------Building the Hierarchical Clustering Model---------------\n"
    #       "------------------------------------------------------------------------")
    app.build_clusters(d)
    # print("------------------------------------------------------------------------\n"
    #       "-------------Iterative Clustering-Based Sampling and Labeling-----------\n"
    #       "------------------------------------------------------------------------")
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d)
        if d.has_ground_truth:
            app.label_with_ground_truth(d)
    # print("------------------------------------------------------------------------\n"
    #       "--------------Propagating User Labels Through the Clusters--------------\n"
    #       "------------------------------------------------------------------------")
    app.propagate_labels(d)
    # print("------------------------------------------------------------------------\n"
    #       "---------------Training and Testing Classification Models---------------\n"
    #       "------------------------------------------------------------------------")
    app.predict_labels(d)
    # if app.SAVE_RESULTS:
    #     print("------------------------------------------------------------------------\n"
    #           "---------------------------Storing the Results--------------------------\n"
    #           "------------------------------------------------------------------------")
    #     app.store_results(d)
    return d.detected_cells
