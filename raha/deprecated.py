
import shutil
import operator
import gzip

@staticmethod
def dataset_profiler():
    """
    Deprecated!
    This method profiles the columns of dataset.
    """
    global d
    if not os.path.exists(os.path.join(d.results_folder, d.name, "strategy-filtering")):
        os.mkdir(os.path.join(d.results_folder, d.name, "strategy-filtering"))
    dp_folder_path = os.path.join(d.results_folder, d.name, "strategy-filtering", "dataset-profiling")
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
    print("The {} dataset is profiled.").format(d.name)


@staticmethod
def evaluation_profiler():
    """
    Deprecated!
    This method computes the performance of the error detection strategies on historical data.
    """
    global d
    ep_folder_path = os.path.join(d.results_folder, d.name, "strategy-filtering", "evaluation-profiling")
    if not os.path.exists(ep_folder_path):
        os.mkdir(ep_folder_path)
    sp_folder_path = os.path.join(d.results_folder, d.name, "strategy-profiling")
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
            #     print ("Performance of {} on {} = {:.2f}, {:.2f}, {:.2f}".format(strategy_name, attribute, precision, recall, f1))
    for j, attribute in enumerate(d.dataframe.columns.tolist()):
        pickle.dump(columns_performance[j], open(os.path.join(ep_folder_path, attribute + ".dictionary"), "wb"))
    print("{} error detection strategies are evaluated.".format(len(strategies_file_list)))


def strategy_filterer(self):
    """
    Deprecated!
    This method uses historical data to rank error detection strategies for the dataset and select the top-ranked.
    """
    global d
    nsp_folder_path = os.path.join(d.results_folder, d.name, "strategy-filtering", "strategy-profiling")
    if not os.path.exists(nsp_folder_path):
        os.mkdir(nsp_folder_path)
    columns_similarity = {}
    for nci, na in enumerate(d.dataframe.columns.tolist()):
        ndp_folder_path = os.path.join(d.results_folder, d.name, "strategy-filtering", "dataset-profiling")
        ncp = pickle.load(open(os.path.join(ndp_folder_path, na + ".dictionary"), "rb"))
        for hdn in self.DATASETS:
            if hdn != d.name:
                hd = dataset.Dataset(self.DATASETS[hdn])
                for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                    hdp_folder_path = os.path.join(d.results_folder, hd.name, "strategy-filtering",
                                                   "dataset-profiling")
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
    print("Column profile similarities are calculated.")
    f1_measure = {}
    for hdn in self.DATASETS:
        if hdn != d.name:
            hd = dataset.Dataset(self.DATASETS[hdn])
            for hci, ha in enumerate(hd.dataframe.columns.tolist()):
                ep_folder_path = os.path.join(d.results_folder, hd.name, "strategy-filtering",
                                              "evaluation-profiling")
                strategies_performance = pickle.load(open(os.path.join(ep_folder_path, ha + ".dictionary"), "rb"))
                if (hd.name, ha) not in f1_measure:
                    f1_measure[(hd.name, ha)] = {}
                for strategy_name in strategies_performance:
                    f1_measure[(hd.name, ha)][strategy_name] = strategies_performance[strategy_name][2]
    print("Previous strategy performances are loaded.")
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
                            if strategy_name not in strategies_score[na] or score >= strategies_score[na][
                                strategy_name]:
                                strategies_score[na][strategy_name] = score
                        elif sn[0] == "regex":
                            sn[1][0][0] = na
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][
                                json.dumps(sn)]:
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
                            if json.dumps(sn) not in strategies_score[na] or score >= strategies_score[na][
                                json.dumps(sn)]:
                                strategies_score[na][json.dumps(sn)] = score
                        else:
                            sys.stderr.write("I do not know this error detection tool!\n")
    print("Strategy scores are calculated.")
    sp_folder_path = os.path.join(d.results_folder, d.name, "strategy-profiling")
    strategies_output = {}
    strategies_runtime = {}
    for strategy_file in os.listdir(sp_folder_path):
        strategy_profile = pickle.load(open(os.path.join(sp_folder_path, strategy_file), "rb"))
        strategies_output[strategy_profile["name"]] = strategy_profile["output"]
        strategies_runtime[strategy_profile["name"]] = strategy_profile["runtime"]
    print("Outputs of the strategies are loaded.")
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








def baselines(self):
    """
    Deprecated!
    This methods implements the baselines.
    """
    global d
    b_folder_path = os.path.join(d.results_folder, d.name, "baselines")
    if not os.path.exists(b_folder_path):
        os.mkdir(b_folder_path)
    sp_folder_path = os.path.join(d.results_folder, d.name, "strategy-profiling")
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
            "patterns": [["id", "^tt[\d]+$", "ONM"], ["year", "^[\d]{4}$", "ONM"],
                         ["rating_value", "^[\d.]*$", "ONM"],
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
        print("dBoost on {} {}".format(d.name, dboost_correction))
        print("Precision = {:.2f}".format(er[0]))
        print("Recall = {:.2f}".format(er[1]))
        print("F1 = {:.2f}".format(er[2]))
        print("--------------------")
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
        print("NADEEF on {}".format(d.name))
        print("Precision = {:.2f}".format(er[0]))
        print("Recall = {:.2f}".format(er[1]))
        print("F1 = {:.2f}".format(er[2]))
        print("--------------------")
    if "KATARA" in self.BASELINES:
        katara_correction = {}
        for strategy_name in strategies_output:
            tool_name = json.loads(strategy_name)[0]
            if tool_name == "katara":
                katara_correction.update({cell: "JUST A DUMMY VALUE" for cell in strategies_output[strategy_name]})
        er = d.evaluate_data_cleaning(katara_correction)[:3]
        pickle.dump(katara_correction.keys(), open(os.path.join(b_folder_path, "katara_output.list"), "wb"))
        print("KATARA on {}".format(d.name))
        print("Precision = {:.2f}".format(er[0]))
        print("Recall = {:.2f}".format(er[1]))
        print("F1 = {:.2f}".format(er[2]))
        print("--------------------")
    if "Min-k" in self.BASELINES:
        fv_folder_path = os.path.join(d.results_folder, d.name, "feature-vectors")
        fv = {j: {(i, j): [] for i in range(d.dataframe.shape[0])} for j in range(d.dataframe.shape[1])}
        for j, attribute in enumerate(d.dataframe.columns.tolist()):
            o_fv, p_fv, r_fv, k_fv = pickle.load(
                gzip.open(os.path.join(fv_folder_path, attribute + ".dictionary"), "rb"))
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
            print("Min-k = {:.2f}".format(k))
            correction_dictionary = {}
            for j in range(d.dataframe.shape[1]):
                for cell in fv[j]:
                    if len(fv[j][cell]) > 0 and float(sum(fv[j][cell])) / len(fv[j][cell]) > k:
                        correction_dictionary[cell] = "JUST A DUMMY VALUE"
            er = d.evaluate_data_cleaning(correction_dictionary)[:3]
            if not max_performance or er[2] > max_performance[2]:
                max_performance = er
                best_k = k
        print("Min-k on {}".format(d.name))
        print("Best k {}".format(best_k))
        print("Precision = {:.2f}".format(max_performance[0]))
        print("Recall = {:.2f}".format(max_performance[1]))
        print("F1 = {:.2f}".format(max_performance[2]))
        print("--------------------")
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
            print("Maximum Entropy on {}".format(d.name))
            print("Labeled Tuples Count = {}".format(len(labeled_tuples)))
            print("Precision = {:.2f}".format(er[0]))
            print("Recall = {:.2f}".format(er[1]))
            print("F1 = {:.2f}".format(er[2]))
            print("--------------------")
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
            print("Run {}".format(r))
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
            print("Metadata Driven on {}".format(d.name))
            print("Labeled Tuples Count = {}".format(s))
            print("Precision = {:.2f}".format(mean_results[s][0]))
            print("Recall = {:.2f}".format(mean_results[s][1]))
            print("F1 = {:.2f}".format(mean_results[s][2]))
            print("--------------------")
            results_string += "({},{:.2f})".format(s, mean_results[s][2])
        results_string += "}; \\addlegendentry{Metadata Driven}"
        print(results_string)
    if "ActiveClean" in self.BASELINES:
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
        text = [" ".join(row) for row in d.dataframe.get_values().tolist()]
        acfv = vectorizer.fit_transform(text).toarray()
        sampling_range = range(1, self.LABELING_BUDGET + 1, 1)
        mean_results = {s: [0.0, 0.0, 0.0] for s in sampling_range}
        # actual_dirty_tuples = {i: 1 for i in range(d.dataframe.shape[0]) if
        #                        int(sum([(i, j) in d.actual_errors_dictionary
        #                        for j in range(d.dataframe.shape[1])]) > 0)}
        for r in range(self.RUN_COUNT):
            print("Run {}".format(r))
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
                    y_train.append(
                        int(sum([(i, j) in d.actual_errors_dictionary for j in range(d.dataframe.shape[1])]) > 0))
                adaptive_detector_output = []
                x_test = [acfv[i, :] for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
                test_rows = [i for i in range(d.dataframe.shape[0]) if i not in labeled_tuples]
                if sum(y_train) == len(y_train):
                    predicted_labels = len(test_rows) * [1]
                elif sum(y_train) == 0 or len(x_train[0]) == 0:
                    predicted_labels = len(test_rows) * [0]
                else:
                    model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1e-6, max_iter=200,
                                                               fit_intercept=True)
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
                mean_results[len(labeled_tuples)] = (
                            numpy.array(mean_results[len(labeled_tuples)]) + numpy.array(er) / self.RUN_COUNT).tolist()
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
            print("ActiveClean on".format(d.name))
            print("Labeled Tuples Count = {}".format(s))
            print("Precision = {:.2f}".format(mean_results[s][0]))
            print("Recall = {:.2f}".format(mean_results[s][1]))
            print("F1 = {:.2f}".format(mean_results[s][2]))
            print("--------------------")
            results_string += "({},{:.2f})".format(s, mean_results[s][2])
        results_string += "}; \\addlegendentry{ActiveClean}"
        print(results_string)
