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
import shutil
import operator
import itertools

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


def dataset_profiler(d):
    """
    This method profiles the columns of dataset.
    """
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


def evaluation_profiler(d):
    """
    This method computes the performance of the error detection strategies on historical data.
    """
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


def get_selected_strategies(self, d, historical_datasets):
    """
    This method uses historical data to rank error detection strategies for the dataset and select the top-ranked.
    """
    nsp_folder_path = os.path.join(d.results_folder, d.name, "strategy-profiling")
    if not os.path.exists(nsp_folder_path):
        os.mkdir(nsp_folder_path)
    columns_similarity = {}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        ndp_folder_path = os.path.join(d.results_folder, d.name, "dataset-profiling")
        ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
        for hdn in historical_datasets:
            if hdn != d.name:
                dataset_dictionary = {
                    "name": hdn,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "clean.csv"))
                }
                hd = raha.dataset.Dataset(dataset_dictionary)
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    hdp_folder_path = os.path.join(d.results_folder, hd.name, "dataset-profiling")
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
    f1_measure = {}
    for hdn in historical_datasets:
        if hdn != d.name:
            dataset_dictionary = {
                "name": hdn,
                "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "dirty.csv")),
                "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "clean.csv"))
            }
            hd = raha.dataset.Dataset(dataset_dictionary)
            for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                ep_folder_path = os.path.join(d.results_folder, hd.name, "evaluation-profiling")
                strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                if (hd.name, ha) not in f1_measure:
                    f1_measure[(hd.name, ha)] = {}
                for strategy_name in strategies_performance:
                    f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
    strategies_score = {a: {} for a in d.dataframe.columns.tolist()}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        for hdn in historical_datasets:
            if hdn != d.name:
                dataset_dictionary = {
                    "name": hdn,
                    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "dirty.csv")),
                    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", hdn, "clean.csv"))
                }
                hd = raha.dataset.Dataset(dataset_dictionary)
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
    sp_folder_path = os.path.join(d.results_folder, "strategy-profiling")
    strategies_output = {}
    strategies_runtime = {}
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
            pickle.dump(strategy_profile, open(
                os.path.join(nsp_folder_path, snd[0] + "-" + str(len(os.listdir(nsp_folder_path))) + ".dictionary"),
                "wb"))
    print("Promising error detection strategies are stored.")
