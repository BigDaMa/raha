import os
import re
import sys
import math
import time
import json
import logging
import random
import pickle
import hashlib
import tempfile
import itertools
import multiprocessing
from multiprocessing import shared_memory as sm

import numpy
import pandas
import scipy.stats
import scipy.spatial
import scipy.cluster
import sklearn.svm
import sklearn.tree
import sklearn.cluster
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.kernel_ridge
import sklearn.neural_network
import sklearn.feature_extraction
from constants import *
import constants

import dask
from distributed import Client, LocalCluster
from dask.distributed import get_worker, get_client
from distributed.worker import Worker
import dask.dataframe
import dataset_parallel as dp

import raha
import container

########################################
class DetectionParallel:
    def __init__(self):
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0
        self.BENCHMARK = True
        self.VERBOSE = False
        self.SAVE_RESULTS = False
        self.CLUSTERING_BASED_SAMPLING = True
        self.STRATEGY_FILTERING = False
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = HOMOGENEITY   # ["homogeneity", "majority"]
        self.ERROR_DETECTION_ALGORITHMS = [OUTLIER_DETECTION, PATTERN_VIOLATION_DETECTION, 
                                           RULE_VIOLATION_DETECTION, KNOWLEDGE_BASE_VIOLATION_DETECTION]
        self.HISTORICAL_DATASETS = []
        self.TFID_ENABLED = False
        self.PRELOADING = False
        self.TIME_TOTAL = 0

    def init_workers(self):
        for worker in Worker._instances:
            worker.shared_df = container.shared_dataframe.read()

    def initialize_dataframe(self, path):
        dataframe = dp.DatasetParallel.read_csv_dataframe(path)
        container.shared_dataframe = dp.SharedDataFrame(dataframe)
        return container.shared_dataframe

    def initialize_dataset(self, dataset_dictionary):
        dataset_par = dp.DatasetParallel(dataset_dictionary)
        dataset_par.labels_per_cluster = {}
        dataset_par.detected_cells = {}
        dataset_par.dataframe_num_rows = container.shared_dataframe.read().shape[0]
        dataset_par.dataframe_num_cols = container.shared_dataframe.read().shape[1]
        dataset_par.create_shared_dataset(dataset_par)

        dirty_dataframe = container.shared_dataframe.read() 
        clean_dataframe = dp.DatasetParallel.read_csv_dataframe(dataset_par.clean_path)
        differences_dict = dp.DatasetParallel.get_dataframes_difference(dirty_dataframe, clean_dataframe)
        #dp.DatasetParallel.create_shared_object(differences_dict, dataset_par.differences_dict_mem_ref)

        return dataset_par, differences_dict
    
    def start_dask_cluster(self, num_workers, logging_level):
        dask.config.set({'distributed.worker.multiprocessing-method': 'fork'})
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, processes=True, memory_limit='100GB', silence_logs=logging_level, dashboard_address=None)
        client = Client(cluster)
        return client

    def cleanup_raha(self, dataset):
        #Clean-Up whole dirty Dataframe.
        container.shared_dataframe.unlink()

        #Clean-Up feature vectors
        for j in range(dataset.dataframe_num_cols):
            dataset.cleanup_object(dataset.dirty_mem_ref + "-feature-result-" + str(j))

        #Clean-Up strategy profiles
        for j in range(dataset.dataframe_num_cols):
            dataset.cleanup_object(dataset.dirty_mem_ref + "-strategy_profiles-col" + str(j))

        #Clean-Up values that were shared for prediction phase(raha)
        dataset.cleanup_object(dataset.own_mem_ref + "-predictvariables")

        #Clean-Up Dataset which was shared while executing raha
        dataset.cleanup_object(dataset.own_mem_ref)


    def run_outlier_strategy(self, configuration, dataset_ref, strategy_name_hash):
        """
        Detects cells which don't match given detection strategy - Outlier Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        worker = get_worker()
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        outputted_cells = {j: {} for j in range(dataset.dataframe_num_cols)}
        dataframe = worker.shared_df
        folder_path = os.path.join(tempfile.gettempdir(), dataset.name +"/")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        dataset_path = os.path.join(tempfile.gettempdir(),dataset.name + "/" + dataset.name + "-" + strategy_name_hash + ".csv")
        #Save Memory by copying prestripped file, read write_csv function
        dataset.write_csv(destination_path=dataset_path, dataframe=dataframe)

        parameters = ["-F", ",", "--statistical", "0.5"] + ["--" + configuration[0]] + configuration[1:] + [dataset_path]
        #print("Worker: " + str(get_worker().id) + " started dboost.run")
        raha.tools.dBoost.dboost.imported_dboost.run(parameters)

        dboost_result_path = dataset_path + "-dboost_output.csv"
        if os.path.exists(dboost_result_path) and os.path.getsize(dboost_result_path) > 0:
            dboost_dataframe = pandas.read_csv(dboost_result_path, sep=",", header=None, encoding="utf-8",
                                               dtype=str, keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())

            for i, j in dboost_dataframe.values.tolist():
                if int(i) > 0:
                    outputted_cells[int(j)][(int(i)-1, int(j))] = ""
            os.remove(dboost_result_path)
               
        os.remove(dataset_path)
        return outputted_cells
        

    def run_pattern_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Pattern Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        worker = get_worker()
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        outputted_cells = {j: {} for j in range(dataset.dataframe_num_cols)}
        column_name, character = configuration
        dataframe = worker.shared_df.loc[:, column_name]
        j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(column_name)
        #print("Worker: " + str(get_worker().id) + " running core run_pattern part")
        for i, value in dataframe.items():
            try:
                if len(re.findall("[" + character + "]", value, re.UNICODE)) > 0:
                    outputted_cells[j][(i, j)] = ""
            except:
                #print("Error occured in run_pattern_strategy in worker  " + str(get_worker().id))
                continue

        return outputted_cells
   
    def run_rule_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Rule Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        worker = get_worker()
        value_dict = {}
        left_attribute, right_attribute = configuration

        #Read Columns as seperate Series Objects - a more memory efficient approach
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        outputted_cells = {j: {} for j in range(dataset.dataframe_num_cols)}
        dataframe_left_column = worker.shared_df.loc[:, left_attribute]
        dataframe_right_column = worker.shared_df.loc[:, right_attribute]

        left_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(left_attribute)
        right_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(right_attribute)

        num_elements = len(dataframe_left_column)
        
        #Process through both columns and use the index to synchronize correct positional access
        for i in numpy.arange(0, num_elements):   
            left_value = dataframe_left_column[i]
            right_value = dataframe_right_column[i]
            if left_value:
                if left_value not in value_dict:
                    value_dict[left_value] = {}
                if right_value:
                    value_dict[left_value][right_value] = 1

        #Update the defect cells dictionary of a cell, if left value references more than 1 right value
        for i in numpy.arange(0, num_elements):
            left_value = dataframe_left_column[i]
            if left_value in value_dict and len(value_dict[left_value]) > 1:
                outputted_cells[left_attribute_j][(i, left_attribute_j)] = ""
                outputted_cells[right_attribute_j][(i, right_attribute_j)] = ""
        
        return outputted_cells

    def run_knowledge_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Knowledge Base Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        outputted_cells_katara = raha.tools.KATARA.katara.run(dataset, configuration)

        outputted_cells = {j: {} for j in range(dataset.dataframe_num_cols)}
        for cell in outputted_cells_katara:
            outputted_cells[cell[1]][cell] = ""
        #print(outputted_cells)
        return outputted_cells

    def parallel_strat_runner_process(self, args):
        """
        Runs all error detection strategies in a seperate worker process.
        """
        start_time = time.time()
        outputted_cells = {}
        dataset_ref, algorithm, configuration = args
        strategy_name = json.dumps([algorithm, configuration])
        strategy_name_hashed = str( int( hashlib.sha1( strategy_name.encode("utf-8")).hexdigest(), base=16))

        match algorithm:
            case constants.OUTLIER_DETECTION:
                #Run outlier detection strategy
                outputted_cells = self.run_outlier_strategy(configuration, dataset_ref, strategy_name_hashed)
            case constants.PATTERN_VIOLATION_DETECTION:
                #Run pattern violation detection strategy
                outputted_cells = self.run_pattern_strategy(configuration, dataset_ref)
            case constants.RULE_VIOLATION_DETECTION:
                #Run rule violation detection strategy
                outputted_cells = self.run_rule_strategy(configuration, dataset_ref)
            case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                #Run knowledge base violation strategy
                outputted_cells = self.run_knowledge_strategy(configuration, dataset_ref)
            case _:
                raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        end_time = time.time()

        dataset = dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        strategy_results = {}
        strategy_results["name"] = strategy_name
        for j in range(dataset.dataframe_num_cols):
            strategy_results["output_col_" + str(j)] = list(outputted_cells[j].keys())
        strategy_results["runtime"] = end_time - start_time

        if self.SAVE_RESULTS:
            dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
            pickle.dump(strategy_results, open(os.path.join(dataset.results_folder, "strategy-profiling", strategy_name_hashed + ".dictionary"), "wb"))
        if self.VERBOSE:
            print("{} cells are detected by {}".format(len(outputted_cells), strategy_name))

        return strategy_results
    
    @staticmethod
    def setup_outlier_metadata(dataset_ref):
        """
        Worker-Process in a parallel manner. Creates Gaussian configuration and Histogram configuration and return them.
        """
        configurations = []
        
        #Create Cartesian Product
        cartesian_config = [
                        list(a) for a in
                        list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                                ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                        list(itertools.product(["gaussian"],
                                                ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]

        configurations.extend([dataset_ref, OUTLIER_DETECTION, conf] for conf in cartesian_config)
        return configurations

    @staticmethod
    def pattern_violation_worker(dataset_ref, column_name):
        """
        Worker-Process in a parallel manner. Extracts all characters of one specific column and returns them.
        """
        worker = get_worker()
        configurations = []
        client = get_client()

        #Load Shared DataFrame column by accessing shared memory area, named by column_name
        dataset_column = worker.shared_df.loc[:, column_name]

        #Concatenate all content of a column into a long string
        column_data = "".join(dataset_column.tolist())
        #Notice which character appeared in this column
        character_dict = {character: 1 for character in column_data} 
        character_dict_list = [[column_name, character] for character in character_dict]  

        configurations.extend([dataset_ref, PATTERN_VIOLATION_DETECTION, conf] for conf in character_dict_list)
        del dataset_column, character_dict, column_data
        return configurations

    @staticmethod
    def setup_pattern_violation_metadata(dataset_ref):
        """
        Calculates Meta-Data for pattern-violation application later on
        """
        futures = []
        configurations = []
        client = get_client()
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_names = dp.DatasetParallel.get_column_names(dataset.dirty_path)
        
        #Call a worker for each column name
        arguments1 = [dataset_ref] * len(column_names)
        arguments2 = [column_name  for column_name in column_names]
        futures.append(client.map(DetectionParallel.pattern_violation_worker, arguments1, arguments2))

        #Gather results and merge "list of lists" into one big list
        results_unmerged = client.gather(futures=futures, direct=True)[0]
        results = list(itertools.chain.from_iterable(results_unmerged))

        return results
    @staticmethod
    def setup_rule_violation_metadata(dataset_ref):
        """
        Calculates Meta-Data for rule-violation application later on.
        This method just creates pairs of column names as potential FDs
        Generates exactly n*(n-1) FD pairs
        """
        configurations = []
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_names = dp.DatasetParallel.get_column_names(dataset.dirty_path)
        column_pairs = [[col1, col2] for (col1,col2) in itertools.product(column_names, column_names) if col1 != col2]
        
        configurations.extend([[dataset_ref, RULE_VIOLATION_DETECTION, column_pair] for column_pair in column_pairs])

        return configurations
    @staticmethod
    def setup_knowledge_violation_metadata(dataset_ref):
        """
        Calculates Meta-Data for knowledge-violation application later on.
        This method just creates the parameters for Katara.
        """
        configurations = []
        paths = [os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base", path) for path in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base"))]
        configurations.extend([[dataset_ref, KNOWLEDGE_BASE_VIOLATION_DETECTION, path] for path in paths])

        return configurations

    def run_strategies(self, dataset):
        """
        Creates strategies metadata and executes each strategy for a seperate worker process
        """
        start_time = time.time()
        strategy_profile_path = os.path.join(dataset.results_folder, "strategy-profiling")
        client = get_client()
        futures = []

        if self.STRATEGY_FILTERING:
            for data_dictionary in self.HISTORICAL_DATASETS + [dataset.dictionary]:
                raha.utilities.dataset_profiler(data_dictionary)
                raha.utilities.evaluation_profiler(data_dictionary)
            return raha.utilities.get_selected_strategies_via_historical_data(dataset.dictionary, self.HISTORICAL_DATASETS)

        if os.path.exists(strategy_profile_path) and self.PRELOADING:
            sys.stderr.write("Preloading strategies' results, as they have already been run on the dataset\n")
            strategy_profiles = [pickle.load(open(os.path.join(strategy_profile_path, strategy_file), "rb"))
                                 for strategy_file in os.listdir(strategy_profile_path)]
            end_time = time.time()
            self.TIME_TOTAL += end_time-start_time
            print("Preloading strategies (parallel): " + str(end_time-start_time))
            return strategy_profiles
        if self.SAVE_RESULTS:
            os.mkdir(strategy_profile_path)
            

        for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
            match algorithm_name:
                case constants.OUTLIER_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_outlier_metadata, dataset.own_mem_ref))
                case constants.PATTERN_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_pattern_violation_metadata, dataset.own_mem_ref))
                case constants.RULE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_rule_violation_metadata, dataset.own_mem_ref))
                case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_knowledge_violation_metadata, dataset.own_mem_ref))
                case _:
                    raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        #Gather Results of all workers, metadata configuration
        results = list(itertools.chain.from_iterable(client.gather(futures=futures, direct=True)))
        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Raha strategy metadata generation(parallel): "+  str(end_time - start_time))

        #Start Detecting Errors in parallel
        futures = client.map(self.parallel_strat_runner_process, results)
        #Gather Results of all workers, detected cells as dicts
        strategy_profiles = client.gather(futures=futures, direct=True)

        for j in range(dataset.dataframe_num_cols):
            strategy_profiles_col = []
            for strategy_profile in strategy_profiles:
                strategy_profiles_col.append(
                {"name": strategy_profile["name"],
                 "output": strategy_profile["output_col_" + str(j)]})
            dp.DatasetParallel.create_shared_object(strategy_profiles_col, dataset.dirty_mem_ref + "-strategy_profiles-col" + str(j))

        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Raha running all strategies total time(parallel): "+  str(end_time - start_time))

        return strategy_profiles


    def generate_features_one_col(self, dataset_ref, column_index):
        """
        Worker-Process. Calculates a feature-matrix for one column. 
        A row represents 1 feature-vector of a cell, the row_index is the x-coordinate of the cell the column_index the y-coordinate.
        A column represents the results of 1 specific strategy on *all* cells.
        Does not return the feature-matrix but rather a reference to it in a shared memory area. 
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        strategy_profiles = dp.DatasetParallel.load_shared_object(dataset.dirty_mem_ref + "-strategy_profiles-col" + str(column_index))
        feature_vectors = numpy.zeros((dataset.dataframe_num_rows, len(strategy_profiles)))

        for strategy_index, strategy_profile in enumerate(strategy_profiles):
            strategy_name = json.loads(strategy_profile["name"])[0]

            if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                for cell in strategy_profile["output"]:
                    if cell[1] == column_index:
                        feature_vectors[cell[0], strategy_index] = 1.0
        if self.TFID_ENABLED:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
            column_name = dp.DatasetParallel.get_column_names(dataset.dirty_path)[column_index]
            corpus = dp.DatasetParallel.load_shared_dataframe(column_name)
            try:
                tfid_features = vectorizer.fit_transform(corpus)
                feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
            except:
                pass

        promising_strategies = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
        feature_vectors = feature_vectors[:, promising_strategies] 

        #Store feature vectors in a shared memory area
        dp.DatasetParallel.create_shared_object(feature_vectors, dataset.dirty_mem_ref + "-feature-result-" + str(column_index))

        return dataset.dirty_mem_ref + "-feature-result-" + str(column_index)

    def generate_features(self, dataset, strategy_profiles):
        """
        Calculates feature vector for each column. A seperate matrix is being built for each column.
        Strategies, which mark all cells as either detected or undetected are being discarded.
        """
        start_time = time.time()
        client = get_client()
        futures = []

        #Start workers and append their future-references to the futures list. From each passed parameter-list one entry is passed to the worker.
        futures.append(client.map(self.generate_features_one_col, [dataset.own_mem_ref] * dataset.dataframe_num_cols, numpy.arange(dataset.dataframe_num_cols)))

        results = client.gather(futures=futures, direct=True)
        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Generate Features(parallel): " + str(end_time - start_time))

        return results

    def build_clusters_single_column(self, dataset_ref, column_index):
        """
        Worker-Process. Calculates all clusters and the respective cells of *one* column.
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_features = dp.DatasetParallel.load_shared_object(dataset.dirty_mem_ref + "-feature-result-" + str(column_index))

        clusters_k_c_ce = {k : {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k : {} for k in range(2, self.LABELING_BUDGET + 2)}

        try:
            clustering_model = scipy.cluster.hierarchy.linkage(column_features, method="average", metric="cosine")
            #The bigger our labeling budget is, the more clusters will be generated per column
            for k in clusters_k_c_ce:
                model_labels = [l-1 for l in scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                #Model label contains a 1D-Array, where each index represents the row of a cell and column_index represents the column of the cell
                #c is the number of the cluster this cell belongs to
                for index, c in enumerate(model_labels):
                        if c not in clusters_k_c_ce[k]:
                            #Create a dict containing all cells which belong to cluster number c
                            #Depends on labeling budget k, which represents the total number of clusters available per column
                            clusters_k_c_ce[k][c] = {}
                        
                                #index = row, column_index = column -> coordinates of a specific cell
                        cell = (index, column_index)
                        clusters_k_c_ce[k][c][cell] = 1
                        cells_clusters_k_ce[k][cell] = c
        except:
            pass
        
        if self.VERBOSE:
            print("A hierarchical clustering model is built for column {}".format(column_index))
        clusters_k_j_c_ce = {k : clusters_k_c_ce[k] for k in range(2, self.LABELING_BUDGET+2)}
        cells_clusters_k_j_ce =  {k : cells_clusters_k_ce[k] for k in range(2, self.LABELING_BUDGET+2)}

        # cells_clusters_j_j_ce[k][j] = clustering_results[j][2][k]
        #print(clusters_k_c_ce if column_index == 0 else "")
        #TODO Think about if you want to return these lists or rather save them in shared mem again.
        #print("\nI'm worker: {}, my task is column {}\nMy result is: {}".format(get_worker().id, column_index, [column_index, clusters_k_c_ce,"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" , cells_clusters_k_ce]))
        return [column_index, clusters_k_j_c_ce, cells_clusters_k_j_ce]

    def build_clusters(self, dataset, features_refs):
        """
        Calculates clusters for all columns.
        """
        start_time = time.time()
        clustering_results = []
        client = get_client()
        futures = []

        futures.append(client.map(self.build_clusters_single_column, [dataset.own_mem_ref]*dataset.dataframe_num_cols, numpy.arange(dataset.dataframe_num_cols)))
        results = client.gather(futures=futures, direct=True)[0]
        results.sort(key= lambda x: x[0], reverse=False)

        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Build clusters (parallel): " + str(end_time-start_time))

        return results
    
    def sample_tuple(self, dataset, clustering_results):
        """
        Calculates a sample-tuple which will later be labeled by the user or by the ground-truth
        """
        k = len(dataset.labeled_tuples)+2
        for j in numpy.arange(dataset.dataframe_num_cols):
            for c in clustering_results[j][1][k]:
                dataset.labels_per_cluster[(j, c)] = {cell : dataset.labeled_cells[cell][0] for cell in clustering_results[j][1][k][c]
                                                             if cell[0] in dataset.labeled_tuples}

        if self.CLUSTERING_BASED_SAMPLING:
            tuple_score = numpy.zeros(dataset.dataframe_num_rows)
            for i in numpy.arange(dataset.dataframe_num_rows):
                if i not in dataset.labeled_tuples:
                    score = 0.0
                    for j in numpy.arange(dataset.dataframe_num_cols):
                        if clustering_results[j][1][k]:
                            cell = (i, j)
                            c = clustering_results[j][2][k][cell]
                            score += math.exp(-len(dataset.labels_per_cluster[(j, c)]))
                    tuple_score[i] = math.exp(score)
        else:
            tuple_score = numpy.ones(dataset.dataframe_num_rows)
        sum_tuple_score = sum(tuple_score)
        p_tuple_score = tuple_score / sum_tuple_score
        dataset.sampled_tuple = numpy.random.choice(numpy.arange(dataset.dataframe_num_rows), 1, p=p_tuple_score)[0]
        if self.VERBOSE:
            print("Tuple {} is sampled".format(dataset.sampled_tuple))

        return dataset.sampled_tuple  
    
    def label_with_ground_truth(self, dataset, differences_dict, clean_dataframe):
        """
        Labels one sampled tuple with ground truth.
        """
        k = len(dataset.labeled_tuples) + 2
        dataset.labeled_tuples[dataset.sampled_tuple] = 1
        actual_errors_dictionary = differences_dict

        for j in numpy.arange(dataset.dataframe_num_cols):
            cell = (dataset.sampled_tuple, j)
            user_label = int(cell in actual_errors_dictionary)
            flip_result_chance = random.random()

            if flip_result_chance > self.USER_LABELING_ACCURACY:
                user_label = 1 - user_label
            dataset.labeled_cells[cell] = [user_label, clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(dataset.sampled_tuple))
        return
    
    def propagate_labels(self, dataset, clustering_results):
        """
        Propagates labels of labeled tuples in their respective cluster depending on the set propagation method.
        """
        start_time = time.time()

        dataset.extended_labeled_cells = {cell: dataset.labeled_cells[cell][0] for cell in dataset.labeled_cells}
        k = len(dataset.labeled_tuples) + 1

        for j in numpy.arange(dataset.dataframe_num_cols):
            cell = (dataset.sampled_tuple, j)
            if cell in clustering_results[j][2][k]:
                c = clustering_results [j][2][k][cell]
                dataset.labels_per_cluster[(j,c)][cell] = dataset.labeled_cells[cell][0]
        
        if self.CLUSTERING_BASED_SAMPLING:
            for j in numpy.arange(dataset.dataframe_num_cols):
                for c in clustering_results[j][1][k]:
                    if len(dataset.labels_per_cluster[(j, c)]) > 0:
                        match self.LABEL_PROPAGATION_METHOD:
                            case constants.HOMOGENEITY:
                                cluster_label = list(dataset.labels_per_cluster[(j, c)].values())[0]
                                if sum(dataset.labels_per_cluster[(j, c)].values()) in [0, len(dataset.labels_per_cluster[(j, c)])]:
                                    for cell in clustering_results[j][1][k][c]:
                                        dataset.extended_labeled_cells[cell] = cluster_label
                            case constants.MAJORITY:
                                cluster_label = round( sum(dataset.labels_per_cluster[(j, c)].values()) / len(dataset.labels_per_cluster[(j, c)]) )
                                for cell in clustering_results[j][1][k][c]:
                                        dataset.extended_labeled_cells[cell] = cluster_label
                            case _:
                                raise ValueError("The Label Propagation Method" + str(self.LABEL_PROPAGATION_METHOD) +  " is not supported! Try homogeneity or majority.")
        if self.VERBOSE:
            print("The number of labeled data cells increased from {} to {}.".format(len(dataset.labeled_cells), len(dataset.extended_labeled_cells)))
      
        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Propagating labels (parallel): " + str(end_time-start_time))
        return

    @staticmethod
    def predict_labels_single_col(classification_model_name, verbose, dataset_ref, column_index):
        """
        Worker-Process. Trains a classifier for 1 column for erronous cells classification.
        Returns predicted cells of 1 specific column.
        """
        detected_cells_dictionary = {}
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        feature_vectors = dp.DatasetParallel.load_shared_object(dataset.dirty_mem_ref + "-feature-result-" + str(column_index))
        values = dp.DatasetParallel.load_shared_object(dataset.own_mem_ref + "-predictvariables")
        labeled_tuples = values[1]
        extended_labeled_cells = values[0]

        x_train = [feature_vectors[i, :] for i in numpy.arange(dataset.dataframe_num_rows) if (i, column_index) in extended_labeled_cells]
        y_train = [extended_labeled_cells[(i, column_index)] for i in numpy.arange(dataset.dataframe_num_rows) if (i, column_index) in extended_labeled_cells]
        x_test = feature_vectors

        #Check if all cells are 1's
        if sum(y_train) == len(y_train):
            predicted_labels = numpy.ones(dataset.dataframe_num_rows)
        #Check if all cells are 0's 
        elif sum(y_train) == 0 or len(x_train[0]) == 0:
            predicted_labels = numpy.zeros(dataset.dataframe_num_rows)
        else:
            match classification_model_name:
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
                    raise ValueError("Classification Model Name " + str(classification_model_name) + " is not supported!")
            classification_model.fit(x_train, y_train)
            predicted_labels = classification_model.predict(x_test)

        for i, predicted_label in enumerate(predicted_labels):
            if (i in labeled_tuples and extended_labeled_cells[(i, column_index)]) or (i not in labeled_tuples and predicted_label):
                detected_cells_dictionary[(i, column_index)] = "1"
        if verbose:
            print("A classifier is trained and applied on column {}.".format(column_index))
        return detected_cells_dictionary


    def predict_labels(self, dataset, clustering_results):
        """
        Predicts labels of unclustered cells in a parallel manner.
        Returns all predicted cells of each column.
        """
        start_time = time.time()
        client = get_client()
        futures = []
        column_count = dataset.dataframe_num_cols
        dp.DatasetParallel.create_shared_object([dataset.extended_labeled_cells, dataset.labeled_tuples], dataset.own_mem_ref + "-predictvariables")

        futures.append(client.map(
            DetectionParallel.predict_labels_single_col,
            [self.CLASSIFICATION_MODEL]*column_count, [self.VERBOSE]*column_count, [dataset.own_mem_ref]*column_count, numpy.arange(column_count))
            )
        results = client.gather(futures=futures, direct=True)[0]
        dataset.detected_cells = {cell: "JUST A DUMMY VALUE" for result in results for cell in result}


        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Predict (parallel): " +str(end_time-start_time))

    def run_detection(self, dataset_dictionary):
        #___Initialize DataFrame, Dask Cluster__#
        shared_df = self.initialize_dataframe(dataset_dictionary["path"])

        print("Starting Cluster...")
        client = self.start_dask_cluster(num_workers=os.cpu_count(), logging_level=logging.ERROR)
        client.run(self.init_workers)
        print("Successfully started Cluster.")
        print("Begin Raha Computation")

        #__________Initialize Dataset___________#
        dataset_par, differences_dict = self.initialize_dataset(dataset_dictionary)

        #___________Running Strategies__________#
        strategies = self.run_strategies(dataset_par)

        #__________Generating Features__________#
        self.generate_features(dataset_par, strategies)

        #___________Building Clusters___________#
        clusters = self.build_clusters(dataset_par, [])

        #_______Sampling / Labeling Tuples______#
        start_time = time.time()
        clean_dataframe = dp.DatasetParallel.read_csv_dataframe(dataset_par.clean_path)
        while len(dataset_par.labeled_tuples) < self.LABELING_BUDGET:
            self.sample_tuple(dataset_par, clusters)
            if dataset_par.has_ground_truth:
                self.label_with_ground_truth(dataset_par, differences_dict, clean_dataframe)
        end_time = time.time()
        self.TIME_TOTAL += end_time-start_time
        print("Sampling tuples and labeling with ground truth(parallel): {}".format(end_time-start_time))

        #___________Propagating Labels__________#
        self.propagate_labels(dataset_par, clusters)

        #___________Predicting Labels___________#
        self.predict_labels(dataset_par, clusters)

        print("Raha Detection took {:.3f} seconds in total.".format(self.TIME_TOTAL))
        self.cleanup_raha(dataset_par)
        print(len(dataset_par.detected_cells))
        client.shutdown()
        return dataset_par.detected_cells

########################################
