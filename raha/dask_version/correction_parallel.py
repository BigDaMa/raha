import bz2
import difflib
import itertools
import json
import logging
import math
import os
import pickle
import random
import string
import time
from pathlib import Path

import dask
import dask.dataframe
import numpy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm
import unicodedata
from dask.distributed import get_worker, get_client
from dask.distributed import secede, rejoin
from distributed import Client, LocalCluster
from distributed.worker import Worker

import raha.dask_version.container as container
import raha.dask_version.dataset_parallel as dp
from raha import Correction


class CorrectionParallel(Correction):

    def __init__(self):
        """
        The constructor.
        """
        self.PRETRAINED_VALUE_BASED_MODELS_PATH = ""
        self.VALUE_ENCODINGS = ["identity", "unicode"]
        self.CLASSIFICATION_MODEL = "ABC"  # ["ABC", "DTC", "GBC", "GNB", "KNC" ,"SGDC", "SVC"]
        self.IGNORE_SIGN = "<<<IGNORE_THIS_VALUE>>>"
        self.VERBOSE = False
        self.SAVE_RESULTS = True
        self.ONLINE_PHASE = False
        self.LABELING_BUDGET = 20
        self.MIN_CORRECTION_CANDIDATE_PROBABILITY = 0.0
        self.MIN_CORRECTION_OCCURRENCE = 2
        self.MAX_VALUE_LENGTH = 50
        self.REVISION_WINDOW_SIZE = 5
        self.NUM_WORKERS = os.cpu_count()
        self.CHUNK_SIZE = 100

    def cleanup_baran(self):
        """Deletes all Shared Memory Objects which were allocated while baran was executed."""
        # Clean-Up whole dirty Dataframe.
        container.shared_dataframe.unlink()

        # Clean-Up whole clean Dataframe.
        container.shared_clean_dataframe.unlink()

        # Clean-Up Dataset which was shared while executing baran
        dp.DatasetParallel.cleanup_object("holy_dataset")
        return

    @staticmethod
    def initialize_workers(correct_instance, sampled_tuple=None, dataset_ref=None, step=0):
        """Worker Method. Initializes and Updates all states."""
        if bool(Worker._instances):
            for worker in Worker._instances:
                if step == 0:
                    worker.correct_instance = correct_instance
                    worker.dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
                    worker.dataframe = container.shared_dataframe.read()
                    worker.clean_dataframe = container.shared_clean_dataframe.read()

                worker.dataset.sampled_tuple = sampled_tuple
                worker.correct_instance.label_with_ground_truth(worker.dataset, worker.dataframe,
                                                                worker.clean_dataframe)
                worker.correct_instance.update_models(worker.dataset)
        return

    def initialize_dataframes(self, dictionary):
        """Initializes Global Module DataFrames, which will be shared with Fork+Shared Memory"""
        dataframe = dp.DatasetParallel.read_csv_dataframe(dictionary["path"])
        clean_dataframe = dp.DatasetParallel.read_csv_dataframe(dictionary["clean_path"])
        container.shared_dataframe = dp.SharedDataFrame(dataframe)
        container.shared_clean_dataframe = dp.SharedDataFrame(clean_dataframe)
        return container.shared_dataframe, container.shared_clean_dataframe

    def start_dask_cluster(self, num_workers, logging_level):
        """Starts Dask Cluster."""
        dask.config.set({'distributed.worker.multiprocessing-method': 'fork'})
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, processes=True, memory_limit='100GB',
                               silence_logs=logging_level, dashboard_address=None)
        client = Client(cluster)
        return client

    @staticmethod
    def init_worker_names():
        """Returns Worker names for coordination tasks."""
        if bool(Worker._instances):
            for worker in Worker._instances:
                return worker.name

    @staticmethod
    def initialize_dask(column_errors):
        """Initializes priority system for tasks"""
        column_workers = {}
        client = get_client()
        # print(client)
        names = client.run(CorrectionParallel.init_worker_names)

        # Weight columns to distribute them later on more evenly
        weighted_columns = []
        for column_index in column_errors:
            weighted_columns.append((column_index, len(column_errors[column_index])))
        weighted_columns.sort(key=lambda x: x[1], reverse=True)
        # print(weighted_columns)
        # Distribute in a cycle-direction changing round-robin method

        forward = True
        while (len(weighted_columns) > 0):
            for worker_name in names.values():
                if forward:
                    if len(weighted_columns) != 0:
                        column_workers[weighted_columns.pop(0)[0]] = worker_name
                else:
                    if len(weighted_columns) != 0:
                        column_workers[weighted_columns.pop(-1)[0]] = worker_name
            forward = not forward
        # print(column_workers)
        return column_workers

    def initialize_dataset(self, d):
        """
        Initializes a dataset-object and adds relevant attributes for correction.
        Returns initialized dataset object.
        """
        self.ONLINE_PHASE = True
        dataset = d
        dataset.corrected_cells = {}
        dataset.dataframe_num_rows = container.shared_dataframe.read().shape[0]
        dataset.dataframe_num_cols = container.shared_dataframe.read().shape[1]

        if self.SAVE_RESULTS and not os.path.exists(dataset.results_folder):
            os.mkdir(dataset.results_folder)

        column_errors = {}
        for cell in d.detected_cells:
            self._to_model_adder(column_errors, cell[1], cell)

        # Create Shared Memory Object - The dictionary of column errors.
        dataset.column_errors = column_errors
        dataset.detected_cells = d.detected_cells

        return dataset

    def initialize_models(self, dataset):
        """
        Initializes the error corrector models based on the detected cells.
        """
        if os.path.exists(self.PRETRAINED_VALUE_BASED_MODELS_PATH):
            dataset.value_models = pickle.load(bz2.BZ2FILE(self.PRETRAINED_VALUE_BASED_MODELS_PATH, "rb"))
            if self.VERBOSE:
                print("The pretrained value-based models are loaded.")
        else:
            dataset.value_models = [{}, {}, {}, {}]

        dataframe = container.shared_dataframe.read()
        dataset.domain_models = {}
        num_cols = dataset.dataframe_num_cols

        # Initialize vicinity models. Represents functional dependencies, where j_1 -> j_2
        dataset.vicinity_models = {j_1: {j_2: {} for j_2 in range(num_cols)} for j_1 in range(num_cols)}

        for row in dataframe.itertuples():
            i = row[0]
            row = row[1:]
            vicinity_list = [clean_cell_value if (i, j) not in dataset.detected_cells
                             else self.IGNORE_SIGN
                             for j, clean_cell_value in enumerate(row)]
            for j, value in enumerate(row):
                if (i, j) not in dataset.detected_cells:
                    temp_vicinity_list = list(vicinity_list)
                    temp_vicinity_list[j] = self.IGNORE_SIGN
                    update_dictionary = {
                        "column": j,
                        "new_value": value,
                        "vicinity": temp_vicinity_list
                    }
                    self._vicinity_based_models_updater(dataset.vicinity_models, update_dictionary)
                    self._domain_based_models_updater(dataset.domain_models, update_dictionary)
        if self.VERBOSE:
            print("The error corrector models are initialized.")

    @staticmethod
    def random_string(size):
        """Creates unique, random substring. Used for unique Memory Area Names."""
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for i in range(size))

    @staticmethod
    def _value_encoder(value, encoding):
        """
        This method represents a value with a specified value abstraction encoding method.
        """
        if encoding == "identity":
            return json.dumps(list(value))
        if encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])

    @staticmethod
    def _to_model_adder(model, key, value):
        """
        Adds a key-value pair into a dictionary implemented model.
        """
        if key not in model:
            model[key] = {}
        if value not in model[key]:
            model[key][value] = 0.0
        model[key][value] += 1.0

    def _domain_based_models_updater(self, model, update_dictionary):
        """
        Updates the domain-based error corrector model with a given update dictionary.
        """
        # Note how often a corrected_value appears in a given column. The more often a clean value appears,
        # the more relevant it is, thus being a potential correction candidate for erronous values in the same column.
        self._to_model_adder(model, update_dictionary["column"], update_dictionary["new_value"])

    def _vicinity_based_models_updater(self, models, update_dictionary):
        """
        Updates the vicinity-based error corrector models with a given update dictionary.
        """
        for j_1, cell_value in enumerate(update_dictionary["vicinity"]):
            # If the value is a clean value, we store how often the clean value of column j_1 in the *same* row points to the other clean value of column j_2
            # in the *same* row. Meaning we noted a value pair for a functional dependency j_1->j_2 and also a "correct dependency"
            if cell_value != self.IGNORE_SIGN:
                j_2 = update_dictionary["column"]
                self._to_model_adder(models[j_1][j_2], cell_value, update_dictionary["new_value"])

    def _value_based_models_updater(self, models, update_dictionary):
        """
        Updates the value-based error corrector models with a given update dictionary.
        """
        # TODO: adding jabeja konannde bakhshahye substring
        if self.ONLINE_PHASE or (
                update_dictionary["new_value"] and len(update_dictionary["new_value"]) <= self.MAX_VALUE_LENGTH and
                update_dictionary["old_value"] and len(update_dictionary["old_value"]) <= self.MAX_VALUE_LENGTH and
                update_dictionary["old_value"] != update_dictionary["new_value"] and
                update_dictionary["old_value"].lower() != "n/a" and
                not update_dictionary["old_value"][0].isdigit()):
            remover_transformation = {}
            adder_transformation = {}
            replacer_transformation = {}
            s = difflib.SequenceMatcher(None, update_dictionary["old_value"], update_dictionary["new_value"])
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                index_range = json.dumps([i1, i2])
                if tag == "delete":
                    remover_transformation[index_range] = ""
                if tag == "insert":
                    adder_transformation[index_range] = update_dictionary["new_value"][j1:j2]
                if tag == "replace":
                    replacer_transformation[index_range] = update_dictionary["new_value"][j1:j2]
            for encoding in self.VALUE_ENCODINGS:
                encoded_old_value = self._value_encoder(update_dictionary["old_value"], encoding)
                if remover_transformation:
                    self._to_model_adder(models[0], encoded_old_value, json.dumps(remover_transformation))
                if adder_transformation:
                    self._to_model_adder(models[1], encoded_old_value, json.dumps(adder_transformation))
                if replacer_transformation:
                    self._to_model_adder(models[2], encoded_old_value, json.dumps(replacer_transformation))
                self._to_model_adder(models[3], encoded_old_value, update_dictionary["new_value"])

    def _value_based_corrector(self, models, ed):
        """
        This method takes the value-based models and an error dictionary to generate potential value-based corrections.
        """
        results_list = []
        for m, model_name in enumerate(["remover", "adder", "replacer", "swapper"]):
            model = models[m]
            for encoding in self.VALUE_ENCODINGS:
                results_dictionary = {}
                encoded_value_string = self._value_encoder(ed["old_value"], encoding)
                if encoded_value_string in model:
                    sum_scores = sum(model[encoded_value_string].values())
                    if model_name in ["remover", "adder", "replacer"]:
                        for transformation_string in model[encoded_value_string]:
                            index_character_dictionary = {i: c for i, c in enumerate(ed["old_value"])}
                            transformation = json.loads(transformation_string)
                            for change_range_string in transformation:
                                change_range = json.loads(change_range_string)
                                if model_name in ["remover", "replacer"]:
                                    for i in range(change_range[0], change_range[1]):
                                        index_character_dictionary[i] = ""
                                if model_name in ["adder", "replacer"]:
                                    ov = "" if change_range[0] not in index_character_dictionary else \
                                        index_character_dictionary[change_range[0]]
                                    index_character_dictionary[change_range[0]] = transformation[
                                                                                      change_range_string] + ov
                            new_value = ""
                            for i in range(len(index_character_dictionary)):
                                new_value += index_character_dictionary[i]
                            pr = model[encoded_value_string][transformation_string] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                    if model_name == "swapper":
                        for new_value in model[encoded_value_string]:
                            pr = model[encoded_value_string][new_value] / sum_scores
                            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                                results_dictionary[new_value] = pr
                results_list.append(results_dictionary)
        return results_list

    def _vicinity_based_corrector(self, models, ed):
        """
        This method takes the vicinity-based models and an error dictionary to generate potential vicinity-based corrections.
        """
        results_list = []
        for j, cv in enumerate(ed["vicinity"]):
            results_dictionary = {}
            if j != ed["column"] and cv in models[j][ed["column"]]:
                sum_scores = sum(models[j][ed["column"]][cv].values())
                for new_value in models[j][ed["column"]][cv]:
                    pr = models[j][ed["column"]][cv][new_value] / sum_scores
                    if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                        results_dictionary[new_value] = pr
            results_list.append(results_dictionary)
        return results_list

    def _domain_based_corrector(self, model, ed):
        """
        This method takes a domain-based model and an error dictionary to generate potential domain-based corrections.
        """
        results_dictionary = {}
        sum_scores = sum(model[ed["column"]].values())
        for new_value in model[ed["column"]]:
            pr = model[ed["column"]][new_value] / sum_scores
            if pr >= self.MIN_CORRECTION_CANDIDATE_PROBABILITY:
                results_dictionary[new_value] = pr
        return [results_dictionary]

    def sample_tuple(self, dataset):
        """
        Samples a tuple with random choice from a given pool of good tuple candidates.
        """
        remaining_column_erroneous_cells = {}
        remaining_column_erroneous_values = {}
        column_errors = dataset.column_errors
        dataframe = container.shared_dataframe.read()

        # Create two dicts, the first one contains all erroneous cells of a given column. The second one contains all erroneus values of a column
        for j in column_errors:
            for cell in column_errors[j]:
                if cell not in dataset.corrected_cells:
                    self._to_model_adder(remaining_column_erroneous_cells, j, cell)
                    self._to_model_adder(remaining_column_erroneous_values, j, dataframe.iloc[cell])

        tuple_score = numpy.ones(dataset.dataframe_num_rows)

        tuple_score[list(dataset.labeled_tuples.keys())] = 0.0

        for j in remaining_column_erroneous_cells:
            for cell in remaining_column_erroneous_cells[j]:
                value = dataframe.iloc[cell]
                column_score = math.exp(len(remaining_column_erroneous_cells[j]) / len(column_errors[j]))
                cell_score = math.exp(
                    remaining_column_erroneous_values[j][value] / len(remaining_column_erroneous_cells[j]))
                tuple_score[cell[0]] *= column_score * cell_score
        dataset.sampled_tuple = numpy.random.choice(numpy.argwhere(tuple_score == numpy.amax(tuple_score)).flatten())

        if self.VERBOSE:
            print("Tuple {} is sampled".format(dataset.sampled_tuple))

    def label_with_ground_truth(self, dataset, dataframe, clean_dataframe):
        """
        Labels a tuple with ground truth. If there is no ground truth, an interactive labeling is possible with the jupiter notebook.
        """

        dataset.labeled_tuples[dataset.sampled_tuple] = 1
        for j in numpy.arange(dataset.dataframe_num_cols):
            # A dataset.sampled_tuple is just a the index of the sampled row.
            cell = (dataset.sampled_tuple, j)
            error_label = 0
            # If the value from ground truth differs from the actual value, then we note the label as 1
            if dataframe.iloc[cell] != clean_dataframe.iloc[cell]:
                error_label = 1
            dataset.labeled_cells[cell] = [error_label, clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} has been labeled.".format(dataset.sampled_tuple))

    def update_models(self, dataset):
        """
        Updates the error corrector models with a new labeled tuple.
        """
        detected_cells = dataset.detected_cells
        column_errors = dataset.column_errors
        dataframe = container.shared_dataframe.read()

        # Store all cell values of currently labeled, sampled tuple
        cleaned_sampled_tuple = [dataset.labeled_cells[(dataset.sampled_tuple, j)][1] for j in
                                 numpy.arange(dataset.dataframe_num_cols)]

        for j in numpy.arange(dataset.dataframe_num_cols):
            cell = (dataset.sampled_tuple, j)
            update_dictionary = {
                "column": cell[1],
                "old_value": dataframe.iloc[cell],
                "new_value": cleaned_sampled_tuple[j]
            }

            cell_label = dataset.labeled_cells[cell][0]
            if cell_label == 1:
                # Cell was erroneous
                if cell not in detected_cells:
                    detected_cells[cell] = self.IGNORE_SIGN
                    self._to_model_adder(column_errors, cell[1], cell)
                self._value_based_models_updater(dataset.value_models, update_dictionary)
                self._domain_based_models_updater(dataset.domain_models, update_dictionary)
                # Take new vicinities with cleaned values
                update_dictionary["vicinity"] = [cv if j != cj else self.IGNORE_SIGN
                                                 for cj, cv in enumerate(cleaned_sampled_tuple)]
            else:
                # Cell was not erroneus
                # Take new vicinities with cleaned values, only from those which were labeled dirty to begin with.
                update_dictionary["vicinity"] = [
                    cv if j != cj and dataset.labeled_cells[(dataset.sampled_tuple, cj)][0] == 1
                    else self.IGNORE_SIGN for cj, cv in enumerate(cleaned_sampled_tuple)]
            self._vicinity_based_models_updater(dataset.vicinity_models, update_dictionary)
        if self.VERBOSE:
            print("The error corrector models are updated with new labeled tuple {}.".format(dataset.sampled_tuple))

    def generate_features(self, dataset, cell):
        vicinity = container.shared_dataframe.read().iloc[cell[0], :]
        """Generates all features for one specific cell."""
        error_dictionary = {"column": cell[1], "old_value": vicinity[cell[1]], "vicinity": vicinity}
        value_corrections = self._value_based_corrector(dataset.value_models, error_dictionary)
        vicinity_corrections = self._vicinity_based_corrector(dataset.vicinity_models, error_dictionary)
        domain_corrections = self._domain_based_corrector(dataset.domain_models, error_dictionary)
        models_corrections = value_corrections + vicinity_corrections + domain_corrections
        corrections_features = {}
        for mi, model in enumerate(models_corrections):
            for correction in model:
                if correction not in corrections_features:
                    corrections_features[correction] = numpy.zeros(len(models_corrections))
                corrections_features[correction][mi] = model[correction]
        return cell, corrections_features

    def generate_features_chunk(self, chunk, step):
        """
        Calculates all corrections for a chunk of cells
        """
        cells_results = []
        worker = get_worker()
        dataset = worker.dataset
        dataframe = worker.dataframe

        cell_x = None
        cell_y = None
        for cell in chunk:
            if cell is not None:
                cell_x = cell[0]
                cell_y = cell[1]
                cells_results.append(self.generate_features(dataset, cell))

        # Create Randomized, unique string, which references the chunk in a completely new shared mem area
        ref = dataset.own_mem_ref + str(step) + self.random_string(10) + str(cell_x) + str(cell_y)
        dp.DatasetParallel.create_shared_object(cells_results, ref)

        return cells_results

    def generate_pair_features_chunk(self, chunk, step):
        """
        Calculates all corrections for a chunk of cells
        """
        cells_results = []
        pair_features = {}
        pairs_counter = 0
        worker = get_worker()
        dataset = worker.dataset
        dataframe = worker.dataframe

        cell_x = None
        cell_y = None
        for cell in chunk:
            if cell is not None:
                cell_x = cell[0]
                cell_y = cell[1]
                cells_results.append(self.generate_features(dataset, cell))

        for cell, corrections_features in cells_results:
            pair_features[cell] = {}
            for correction in corrections_features:
                pair_features[cell][correction] = corrections_features[correction]
                pairs_counter += 1

        return pair_features

    def summarize_features_one_col(self, chunk_results):
        """Collects all features for one column"""
        pairs_counter = 0
        pair_features = {}
        chunk_results_unpacked = list(itertools.chain.from_iterable(chunk_results))

        for cell, corrections_features in chunk_results_unpacked:
            pair_features[cell] = {}
            for correction in corrections_features:
                pair_features[cell][correction] = corrections_features[correction]
                pairs_counter += 1
        if self.VERBOSE:
            print("{} pairs of (a data error, a potential correction) are featurized.".format(pairs_counter))

        return pair_features

    def generate_and_predict(self, column_workers, column_errors, step):
        """
        Generates the tasks for predicting and generating features. For each column
        the erroneous cells will be chunked, to improve performance in parallel mode.
        Each task calculates exactly one correction set for *one* specific column.
        """
        cols_chunks = {}
        client = get_client()

        for j in column_errors:
            if j not in cols_chunks:
                cols_chunks[j] = []
            for cell in column_errors[j]:
                cols_chunks[j].append(cell)  # Input dataframe.iloc[cell[0], :] instead of 1 for !!!!!

        predict_futures = []
        for column_index in column_workers:
            predict_futures.append(
                client.submit(self.predict_corrections_column, cols_chunks[column_index], column_index, step))
        return predict_futures

    @staticmethod
    def chunk_prediction_process(chunk, mode, classification_model, step):
        """Predicts Corrections for one chunk of cells."""
        worker = get_worker()
        correct_instance = worker.correct_instance
        pair_features = correct_instance.generate_pair_features_chunk(chunk, step)

        corrections = {}

        for cell in pair_features:
            match mode:
                case "normal":
                    predictions = classification_model.predict(list(pair_features[cell].values()))
                case "ones":
                    predictions = numpy.ones(len(pair_features[cell]))
                case "zeroes":
                    return corrections
                case _:
                    raise ValueError("Mode " + str(mode) + " is not supported!")

            cell_correct_candidates = list(pair_features[cell].keys())
            for index, predicted_label in enumerate(predictions):
                if predicted_label:
                    corrections[cell] = cell_correct_candidates[index]

        return corrections

    def predict_corrections_column(self, cells_list, column_index, step):
        """
        Generates features for all cells in one column and predicts corrections afterwards.
        """
        client = get_client()
        worker = get_worker()
        dataset = worker.dataset
        dataframe = worker.dataframe
        column_errors = worker.dataset.column_errors

        cells_train = []
        cells_test = []
        for cell in cells_list:
            if cell in dataset.detected_cells:
                if cell in dataset.labeled_cells and dataset.labeled_cells[cell][0] == 1:
                    cells_train.append(cell)
                else:
                    cells_test.append(cell)

        pair_features_train = self.generate_pair_features_chunk(cells_train, step)
        corrected_cells = {}
        x_train = []
        y_train = []
        test_cell_correction_list = []

        for cell in cells_list:
            if cell in pair_features_train:
                for correction in pair_features_train[cell]:
                    x_train.append(pair_features_train[cell][correction])
                    y_train.append(int(correction == dataset.labeled_cells[cell][1]))
                    corrected_cells[cell] = dataset.labeled_cells[cell][1]
        if x_train:
            match self.CLASSIFICATION_MODEL:
                case "ABC":
                    classification_model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
                case "DTC":
                    classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini")
                case "GBC":
                    classification_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100)
                case "GNB":
                    classification_model = sklearn.naive_bayes.GaussianNB()
                case "KNC":
                    classification_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
                case "SGDC":
                    classification_model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
                case "SVC":
                    classification_model = sklearn.svm.SVC(kernel="sigmoid")
                case _:
                    raise ValueError("Classification Model " + str(self.CLASSIFICATION_MODEL) + " is not supported!")

            mode = "normal"
            if sum(y_train) == 0:
                mode = "zeroes"
            elif sum(y_train) == len(y_train):
                mode = "ones"
            else:
                classification_model.fit(x_train, y_train)

            chunks = list(itertools.zip_longest(*[iter(cells_test)] * self.CHUNK_SIZE))
            # Todo write that chunk function, collect results, clean them and return
            pair_feature_chunk_refs = client.map(CorrectionParallel.chunk_prediction_process, chunks,
                                                 [mode] * len(chunks), [classification_model] * len(chunks),
                                                 [step] * len(chunks))

            secede()
            corrected_cells_chunks = client.gather(futures=pair_feature_chunk_refs, direct=True)
            rejoin()

            for corrected_chunk in corrected_cells_chunks:
                corrected_cells.update(corrected_chunk)

        if self.VERBOSE:
            print("Column {}, corrected {} cells".format(column_index, len(worker.corrected_cells[column_index])))

        return corrected_cells

    def collect_corrections(self, column_prediction_futures, corrected_cells):
        """
        Collects all corrections for all columns, which were calculated in the current iteration step.
        """
        client = get_client()
        corrected_cells_cols = client.gather(futures=column_prediction_futures, direct=True)

        for corrected_cells_col in corrected_cells_cols:
            corrected_cells.update(corrected_cells_col)
        return corrected_cells

    def predict_corrections(self, d):
        """
        Collects all corrections for all columns, which were calculated in the current iteration step.
        """
        client = get_client()
        corrected_cells_cols = client.gather(futures=d.column_prediction_futures, direct=True)

        for corrected_cells_col in corrected_cells_cols:
            d.corrected_cells.update(corrected_cells_col)
        return d.corrected_cells

    def run(self, d):
        shared_df, clean_df = self.initialize_dataframes(dataset_dictionary)
        client = self.start_dask_cluster(num_workers=os.cpu_count(), logging_level=logging.ERROR)

        # ____________________Initialize Dataset____________________#
        dataframe = shared_df.read()
        clean_dataframe = clean_df.read()
        dataset = self.initialize_dataset(d)
        column_workers = self.initialize_dask(dataset.column_errors)
        # ____________________Initialize Models_____________________#
        self.initialize_models(dataset)

        # __________________Sample, Label and Learn_________________#
        step = 0
        time_sum = 0
        corrected_cells = dataset.corrected_cells
        dp.DatasetParallel.create_shared_object(dataset, "holy_dataset")

        while len(dataset.labeled_tuples) < self.LABELING_BUDGET:
            start_time = time.time()

            # --------------Sample, Label and Update Models--------------#
            self.sample_tuple(dataset)
            if dataset.has_ground_truth:
                self.label_with_ground_truth(dataset, dataframe, clean_dataframe)
            # else label interactively with the jupyter notebook.

            self.update_models(dataset)
            end_time = time.time()
            client.run(self.initialize_workers, correct_instance=self, dataset_ref="holy_dataset",
                       sampled_tuple=dataset.sampled_tuple, step=step)

            # --------------Start Generating and Predicting--------------#
            dataset.column_prediction_futures = self.generate_and_predict(column_workers, dataset.column_errors, step)
            self.predict_corrections(dataset)

            step += 1
            end_time = time.time()
            time_sum += end_time - start_time
            if self.VERBOSE:
                print("PARALLEL step {}: {}".format(step, end_time - start_time))

            # ------------------------------------------------------------#
        self.cleanup_baran()

        # This is necessary to catch errors that sometime occur while shutting down the client.
        # Those errors can be safely ignored
        try:
            client.shutdown()
        except Exception as e:
            pass

        # print(list(dataset.labeled_tuples.keys()))
        if self.VERBOSE:
            print("Total time PARALLEL: {}".format(time_sum))
        return dataset.corrected_cells


########################################

########################################
if __name__ == '__main__':
    dataset_dictionary = {
        "name": "flights",
        "path": str(Path("./datasets/flights/dirty.csv").resolve()),
        "clean_path": str(Path("./datasets/flights/clean.csv").resolve()),
    }
    dataset = dp.DatasetParallel(dataset_dictionary)
    dataset.detected_cells = dict(dataset.get_actual_errors_dictionary())
    print("Detected {} cells!".format(len(dataset.detected_cells)))
    print("________________")

    print("________________")
    print("Running Baran...\n")
    baran = CorrectionParallel()
    corrected_cells = baran.run(dataset)
    print("Corrected {} cells!".format(len(corrected_cells)))
    print("________________")

    p, r, f = dataset.get_data_cleaning_evaluation(corrected_cells)[-3:]
    print(
        "Total Performance on Data-Cleaning {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(dataset.name,
                                                                                                          p, r, f))
    # Sometimes the futures throw an exception. Can be safely ignored
    # https://github.com/dask/distributed/issues/4305
