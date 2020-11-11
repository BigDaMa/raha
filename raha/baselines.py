########################################
# Baselines
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# April 2018
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import re
import json
import random
import pickle
import operator
import itertools

import numpy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.feature_extraction

import raha
########################################


########################################
class Baselines:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.VERBOSE = False
        self.DATASET_CONSTRAINTS = {
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
            "rayyan": {
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
                "patterns": [["id", "^tt[\d]+$", "ONM"], ["year", "^[\d]{4}$", "ONM"],
                             ["rating_value", "^[\d.]*$", "ONM"],
                             # ["rating_count", "^[\d]*$", "ONM"],
                             ["duration", "^([\d]+[ ]min)*$", "ONM"]]
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

    def run_dboost(self, dd):
        """
        This method runs dBoost.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "------------------------------Running dBoost----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:int(d.dataframe.shape[0] / 100.0)]}
        best_f1 = -1.0
        best_strategy = ""
        detection_dictionary = {}
        for strategy_profile in strategy_profiles_list:
            algorithm = json.loads(strategy_profile["name"])[0]
            if algorithm == "OD":
                strategy_output = {cell: "JUST A DUUMY VALUE" for cell in strategy_profile["output"]}
                er = d.get_data_cleaning_evaluation(strategy_output, sampled_rows_dictionary=labeled_tuples)[:3]
                if er[2] > best_f1:
                    best_f1 = er[2]
                    best_strategy = strategy_profile["name"]
                    detection_dictionary = dict(strategy_output)
        return detection_dictionary

    def run_nadeef(self, dd):
        """
        This method runs NADEEF.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "------------------------------Running NADEEF----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        detection_dictionary = {}
        for fd in self.DATASET_CONSTRAINTS[d.name]["functions"]:
            l_attribute, r_attribute = fd
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
                    detection_dictionary[(i, l_j)] = "JUST A DUUMY VALUE"
                    detection_dictionary[(i, r_j)] = "JUST A DUUMY VALUE"
        for attribute, pattern, opcode in self.DATASET_CONSTRAINTS[d.name]["patterns"]:
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].iteritems():
                if opcode == "OM":
                    if len(re.findall(pattern, value, re.UNICODE)) > 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                else:
                    if len(re.findall(pattern, value, re.UNICODE)) == 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
        return detection_dictionary

    def run_katara(self, dd):
        """
        This method runs KATARA.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "------------------------------Running KATARA----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        detection_dictionary = {}
        for strategy_profile in strategy_profiles_list:
            algorithm = json.loads(strategy_profile["name"])[0]
            if algorithm == "KBVD":
                detection_dictionary.update({cell: "JUST A DUUMY VALUE" for cell in strategy_profile["output"]})
        return detection_dictionary

    def run_activeclean(self, dd, sampling_budget=20):
        """
        This method runs ActiveClean.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "----------------------------Running ActiveClean-------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
        text = [" ".join(row) for row in d.dataframe.values.tolist()]
        acfv = vectorizer.fit_transform(text).toarray()
        labeled_tuples = {}
        adaptive_detector_output = []
        detection_dictionary = {}
        while len(labeled_tuples) < sampling_budget:
            if len(adaptive_detector_output) < 1:
                adaptive_detector_output = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
            labeled_tuples.update({i: 1 for i in numpy.random.choice(adaptive_detector_output, 1, replace=False)})
            x_train = []
            y_train = []
            for i in labeled_tuples:
                x_train.append(acfv[i, :])
                y_train.append(int(sum([(i, j) in actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0))
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
            detection_dictionary = {}
            for index, pl in enumerate(predicted_labels):
                i = test_rows[index]
                if pl:
                    adaptive_detector_output.append(i)
                    for j in range(d.dataframe.shape[1]):
                        detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
            for i in labeled_tuples:
                for j in range(d.dataframe.shape[1]):
                    detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
        return detection_dictionary

    def run_min_k(self, dd):
        """
        This method runs min-k
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "------------------------------Running Min-k-----------------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        cells_counter = {}
        for strategy_profile in strategy_profiles_list:
            for cell in strategy_profile["output"]:
                if cell not in cells_counter:
                    cells_counter[cell] = 0.0
                cells_counter[cell] += 1.0
        for cell in cells_counter:
            cells_counter[cell] /= len(strategy_profiles_list)
        thresholds_list = [0.0, 0.2, 0.4, 0.6, 0.8]
        detection_dictionary = {}
        best_f1 = 0.0
        for k in thresholds_list:
            temp_output = {}
            for cell in cells_counter:
                if cells_counter[cell] >= k:
                    temp_output[cell] = "JUST A DUMMY VALUE"
            er = d.get_data_cleaning_evaluation(temp_output)[:3]
            if er[2] > best_f1:
                best_f1 = er[2]
                detection_dictionary = dict(temp_output)
        return detection_dictionary

    def run_maximum_entropy(self, dd, sampling_budget=20):
        """
        This method runs maximum entropy.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------------Running Maximum Entropy-----------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        sp_folder_path = os.path.join(os.path.dirname(dd["path"]), "raha-baran-results-" + d.name, "strategy-profiling")
        strategy_profiles_list = [pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
                                  for strategy_file in os.listdir(sp_folder_path)]
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:10]}
        detection_dictionary = {}
        while len(labeled_tuples) < sampling_budget:
            best_precision = -1.0
            best_strategy_index = 0
            for strategy_index, strategy_profile in enumerate(list(strategy_profiles_list)):
                tp = 0.0
                for cell in strategy_profile["output"]:
                    if cell in actual_errors_dictionary:
                        tp += 1
                precision = 0.0 if len(strategy_profile["output"]) == 0 else tp / len(strategy_profile["output"])
                if precision > best_precision:
                    best_precision = precision
                    best_strategy_index = strategy_index
            for cell in strategy_profiles_list[best_strategy_index]["output"]:
                detection_dictionary[cell] = "JUST A DUMMY VALUE"
                labeled_tuples[cell[0]] = 1
            strategy_profiles_list.pop(best_strategy_index)
        return detection_dictionary

    def run_metadata_driven(self, dd, sampling_budget=20):
        """
        This method runs metadata driven.
        """
        if self.VERBOSE:
            print("------------------------------------------------------------------------\n"
                  "--------------------------Running Metadata Driven-----------------------\n"
                  "------------------------------------------------------------------------")
        d = raha.dataset.Dataset(dd)
        actual_errors_dictionary = d.get_actual_errors_dictionary()
        dboost_output = self.run_dboost(dd)
        nadeef_output = self.run_nadeef(dd)
        katara_output = self.run_katara(dd)
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
            for la, ra in self.DATASET_CONSTRAINTS[d.name]["functions"]:
                lfv[cell] += [1 if d.dataframe.columns.tolist()[cell[1]] in [la, ra] else 0]
        random_tuples_list = [i for i in random.sample(range(d.dataframe.shape[0]), d.dataframe.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:sampling_budget]}
        x_train = []
        y_train = []
        for cell in cells_list:
            if cell[0] in labeled_tuples:
                x_train.append(lfv[cell])
                y_train.append(int(cell in actual_errors_dictionary))
        detection_dictionary = {}
        if sum(y_train) != 0:
            x_test = [lfv[cell] for cell in cells_list]
            test_cells = [cell for cell in cells_list]
            if sum(y_train) != len(y_train):
                model = sklearn.ensemble.AdaBoostClassifier(n_estimators=6)
                model.fit(x_train, y_train)
                predicted_labels = model.predict(x_test)
            else:
                predicted_labels = len(test_cells) * [1]
            detection_dictionary = {}
            for index, pl in enumerate(predicted_labels):
                cell = test_cells[index]
                if cell[0] in labeled_tuples:
                    if cell in actual_errors_dictionary:
                        detection_dictionary[cell] = "JUST A DUMMY VALUE"
                elif pl:
                    detection_dictionary[cell] = "JUST A DUMMY VALUE"
        return detection_dictionary
########################################


########################################
if __name__ == "__main__":
    dataset_name = "hospital"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    app = Baselines()
    app.run_dboost(dataset_dictionary)
########################################
