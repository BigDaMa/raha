########################################
# Dataset
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import re
import os
import sys
import html
import shutil

import numpy
import pandas
import dask
import dask.dataframe as dd
import raha.constants as constants

import pickle
from multiprocessing import shared_memory as sm
from dask.distributed import get_client
import hashlib
from raha.dataset import Dataset as Data


########################################


########################################
class DatasetParallel(Data):
    """
    The dataset class.
    """

    def __init__(self, dataset_dictionary):
        """
        The constructor creates a dataset.
        """
        self.name = dataset_dictionary["name"]
        self.own_mem_ref = self.hash_with_salt(constants.DATASET_MEMORY_REF)[:5]

        self.dirty_mem_ref = self.hash_with_salt(dataset_dictionary["name"])[:5]
        self.clean_mem_ref = self.hash_with_salt(dataset_dictionary["name"] + '-clean')[:5]
        self.differences_dict_mem_ref = self.dirty_mem_ref + "-dif-d"
        self.dirty_path = dataset_dictionary["path"]
        self.dictionary = dataset_dictionary
        self.dataframe_num_rows = 0
        self.dataframe_num_cols = 0
        self.labeled_tuples = {}
        self.labeled_cells = {}
        self.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]),
                                           "raha-baran-results-" + dataset_dictionary["name"])

        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

        if "clean_path" in dataset_dictionary:
            self.has_ground_truth = True
            self.clean_path = dataset_dictionary["clean_path"]
        if "repaired_path" in dataset_dictionary:
            self.has_been_repaired = True
            self.repaired_path = dataset_dictionary["repaired_path"]

    def initialize_dataset(self, create_frame=True, create_split=True, create_clean=True, create_dataset=True,
                           create_diffs=True):
        """
        Creates Shared-Memory areas and loads the corresponding dataframe into it.
        For each column one area is created, also one for the whole dataframe with all columns
        Stores its own object into shared memory
        """
        if create_frame:
            self.create_shared_dataframe(self.dirty_path, self.dirty_mem_ref, dataset=self)
        if create_clean:
            self.create_shared_dataframe(self.clean_path, self.clean_mem_ref)
        if create_split:
            self.create_shared_split_dataframe(self.dirty_mem_ref)
        if create_dataset:
            self.create_shared_dataset(self)
        if create_frame and create_clean and create_diffs:
            differences_dict = self.get_dataframes_difference(self.load_shared_dataframe(self.dirty_mem_ref),
                                                              self.load_shared_dataframe(self.clean_mem_ref))
            self.create_shared_object(differences_dict, self.differences_dict_mem_ref)

    @staticmethod
    def cleanup_shared_object(name):
        try:
            obj_area = sm.SharedMemory(name=name, create=False)
            obj_area.close()
            obj_area.unlink()
        except Exception as e:
            print(e)

    @staticmethod
    def cleanup_object(name):
        try:
            memory_area = sm.SharedMemory(name=name, create=False)
            memory_area.close()
            memory_area.unlink()
            del memory_area
        except Exception as e:
            print(e)

    @staticmethod
    def create_shared_object(obj, name):
        pickled_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_obj_size = len(pickled_obj)
        shared_mem_area = sm.SharedMemory(name=name, create=True, size=pickled_obj_size)
        shared_mem_area.buf[:pickled_obj_size] = pickled_obj

        shared_mem_area.close()
        del shared_mem_area
        return

    @staticmethod
    def create_shared_dataset(dataset):
        """
        Creates a shared dataset object. The given dataset will be serialized and its output written into a shared memory area.
        Other Processes can obtain this area by referencing the shared memory area by its name.
        """
        pickled_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_dataset_size = len(pickled_dataset)
        shared_mem_area = sm.SharedMemory(name=dataset.own_mem_ref, create=True, size=pickled_dataset_size)
        shared_mem_area.buf[:pickled_dataset_size] = pickled_dataset

        shared_mem_area.close()
        del shared_mem_area
        return

    @staticmethod
    def create_shared_dataframe(dataframe_filepath, mem_area_name, dataset=None):
        """
        Creates a shared dataframe object. The given dataframe will be serialized and its output written into a shared memory area.
        Other Processes can obtain this area by referencing the shared memory area by its name. 
        """
        MB_1 = 1e6
        num_partitions = 10
        client = get_client()
        filesize = os.path.getsize(dataframe_filepath)

        # Aim for 10 partitions
        blocksize = filesize / num_partitions if filesize >= num_partitions else MB_1
        # print("Blocksize of " + dataframe_filepath + " is:" + str(blocksize))

        # Read DataFrame in parallel
        kwargs = {'sep': ',', 'header': 'infer', 'encoding': 'utf-8', 'dtype': str, 'keep_default_na': False,
                  'low_memory': False}
        dataframe = dask.dataframe.read_csv(urlpath=dataframe_filepath, blocksize=int(blocksize), **kwargs).applymap(
            DatasetParallel.value_normalizer)
        dataframe = client.compute(dataframe).result()
        dataframe.reset_index(inplace=True, drop=True)

        if dataset is not None:
            dataset.dataframe_num_rows = dataframe.shape[0]
            dataset.dataframe_num_cols = dataframe.shape[1]

        pickled_dataframe = pickle.dumps(dataframe, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_dataframe_size = len(pickled_dataframe)
        # print("Size of pickled dataframe " + str(pickled_dataframe_size))

        shared_mem_area = sm.SharedMemory(name=mem_area_name, create=True, size=pickled_dataframe_size)
        shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe

        shared_mem_area.close()
        del shared_mem_area
        return mem_area_name

    @staticmethod
    def create_shared_split_dataframe(dataframe_ref):
        """
        Creates several shared memory areas. Each area contains a single column of a given dataframe as pandas.Series objects.
        The given dataframes will be serialized and its output written into the corresponding shared memory area.
        Other Processes can obtain these areas by referencing the shared memory area by its name.
        """
        dataframe = DatasetParallel.load_shared_dataframe(dataframe_ref)

        for column in dataframe.columns.tolist():
            pickled_dataframe = pickle.dumps(dataframe[column], protocol=pickle.HIGHEST_PROTOCOL)
            pickled_dataframe_size = len(pickled_dataframe)
            try:
                shared_mem_area = sm.SharedMemory(name=column, create=True, size=pickled_dataframe_size)
                shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe
                shared_mem_area.close()
            except FileExistsError:
                shared_mem_area = sm.SharedMemory(name=column, create=False, size=pickled_dataframe_size)
                shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe
                shared_mem_area.close()
            except Exception as shm_err:
                print('Failed to create or attach to a split dataframe: {}'.format(shm_err))
                raise

            del shared_mem_area
        return

    @staticmethod
    def load_shared_object(mem_area_name):
        shared_mem_area = sm.SharedMemory(name=mem_area_name, create=False)
        return pickle.loads(shared_mem_area.buf)

    @staticmethod
    def load_shared_num_rows(dataset_ref):
        """
        Loads number of rows from shared memory.
        """
        dataset = DatasetParallel.load_shared_dataset(dataset_ref)
        shared_mem_area = sm.SharedMemory(name=dataset.num_rows_ref, create=False)
        deserialized_num_rows = pickle.loads(shared_mem_area.buf)

        del shared_mem_area
        return deserialized_num_rows

    @staticmethod
    def load_shared_num_cols(dataset_ref):
        """
        Loads number of columns from shared memory.
        """
        dataset = DatasetParallel.load_shared_dataset(dataset_ref)
        shared_mem_area = sm.SharedMemory(name=dataset.num_cols_ref, create=False)
        deserialized_num_cols = pickle.loads(shared_mem_area.buf)

        del shared_mem_area
        return deserialized_num_cols

    @staticmethod
    def load_shared_dataset(dataset_ref):
        """
        Loads a shared memory dataset, which is stored and serialized in a shared_memory area(dataset_ref).
        The loaded dataset will be deserialized and returned.
        """
        shared_mem_area = sm.SharedMemory(name=dataset_ref, create=False)
        deserialized_dataset = pickle.loads(shared_mem_area.buf)

        shared_mem_area.close()
        del shared_mem_area
        return deserialized_dataset

    @staticmethod
    def load_shared_dataframe(dataframe_ref):
        """
        Loads a shared memory dataframe, which is stored and serialized in a shared_memory area(dataframe_ref).
        The loaded dataframe will be deserialized and returned.
        """
        shared_mem_area = sm.SharedMemory(name=dataframe_ref, create=False)
        deserialized_frame = pickle.loads(shared_mem_area.buf)

        shared_mem_area.close()
        del shared_mem_area
        return deserialized_frame

    @staticmethod
    def get_column_names(dataframe_filepath):
        """
        Returns a List of the column names of a given dataframe csv file.
        """
        return pandas.read_csv(dataframe_filepath, nrows=0).columns.tolist()

    @staticmethod
    def read_csv_dataframe(dataframe_path):
        """
        This method reads a dataset from a csv file path.
        """
        # Params to get passed to pandas read_csv function
        dataframe = pandas.read_csv(dataframe_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(DatasetParallel.value_normalizer)
        return dataframe

    @staticmethod
    def write_csv(destination_path, dataframe_ref=None, dataframe=None, copy=False, source_path=None, pickle=False):
        """
        Writes Dataframe as csv file to given path.
        """
        if dataframe_ref is not None:
            DatasetParallel.load_shared_dataframe(dataframe_ref).to_csv(destination_path, sep=",", header=True,
                                                                        index=False, encoding="utf-8")
        elif dataframe is not None:
            dataframe.to_csv(destination_path, sep=",", header=True, index=False, encoding="utf-8")
        else:
            if copy and source_path is not None:
                # print("Copying file from: " + source_path + " to: " + destination_path)
                try:
                    source_path = source_path
                    destination_path = destination_path
                    shutil.copyfile(source_path, destination_path)
                except:
                    raise ValueError("Copying csv to dest failed in write_csv()!")
            else:
                raise ValueError("Not enough values passed in write_csv()!")

    @staticmethod
    def get_dataframes_difference(dataframe_1, dataframe_2):
        """
        This method compares two dataframes and returns the different cells.
        """
        if dataframe_1.shape != dataframe_2.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        difference_dictionary = {}
        difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
        for j in range(dataframe_1.shape[1]):
            for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
                difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
        return difference_dictionary

    def get_actual_errors_dictionary(self):
        clean_df = self.read_csv_dataframe(self.clean_path)
        dirty_df = self.read_csv_dataframe(self.dirty_path)
        return self.get_dataframes_difference(dirty_df, clean_df)

    def create_repaired_dataset(self, correction_dictionary):
        """
        This method takes the dictionary of corrected values and creates the repaired dataset.
        """
        self.repaired_dataframe = self.read_csv_dataframe(self.dirty_path)
        for cell in correction_dictionary:
            self.repaired_dataframe.iloc[cell] = self.value_normalizer(correction_dictionary[cell])

    def get_correction_dictionary(self):
        """
        This method compares the repaired and dirty versions of a dataset.
        """
        return self.get_dataframes_difference(self.read_csv_dataframe(self.dirty_path),
                                              self.read_csv_dataframe(self.clean_path))

    def get_data_quality(self):
        """
        This method calculates data quality of a dataset.
        """
        return (1.0 - float(len(self.get_actual_errors_dictionary())) /
                (self.read_csv_dataframe(self.dirty_path).shape[0] * self.read_csv_dataframe(self.dirty_path).shape[1]))

    def get_data_cleaning_evaluation(self, correction_dictionary, sampled_rows_dictionary=False):
        """
        This method evaluates data cleaning process.
        """
        actual_errors = self.get_actual_errors_dictionary()
        if sampled_rows_dictionary:
            actual_errors = {(i, j): actual_errors[(i, j)] for (i, j) in actual_errors if i in sampled_rows_dictionary}
        ed_tp = 0.0
        ec_tp = 0.0
        output_size = 0.0
        for cell in correction_dictionary:
            if (not sampled_rows_dictionary) or (cell[0] in sampled_rows_dictionary):
                output_size += 1
                if cell in actual_errors:
                    ed_tp += 1.0
                    if correction_dictionary[cell] == actual_errors[cell]:
                        ec_tp += 1.0
        ed_p = 0.0 if output_size == 0 else ed_tp / output_size
        ed_r = 0.0 if len(actual_errors) == 0 else ed_tp / len(actual_errors)
        ed_f = 0.0 if (ed_p + ed_r) == 0.0 else (2 * ed_p * ed_r) / (ed_p + ed_r)
        ec_p = 0.0 if output_size == 0 else ec_tp / output_size
        ec_r = 0.0 if len(actual_errors) == 0 else ec_tp / len(actual_errors)
        ec_f = 0.0 if (ec_p + ec_r) == 0.0 else (2 * ec_p * ec_r) / (ec_p + ec_r)
        return [ed_p, ed_r, ed_f, ec_p, ec_r, ec_f]

    @staticmethod
    def value_normalizer(value):
        """
        This method takes a value and minimally normalizes it.
        """
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value

    def hash_with_salt(self, word):
        salt = os.urandom(16)
        word_encoded = word.encode()
        salted_word_hash = hashlib.sha1(word_encoded + salt).hexdigest()
        return salted_word_hash


class SharedNumpyArray:
    """
    Holds a Numpy Array in Shared Memory.
    """

    def __init__(self, array):
        """
        Stores an initial numpy in Shared-Memory
        """
        # Create Shared Memory Area with the same size as the input array.
        self._shared_buffer = sm.SharedMemory(create=True, size=array.nbytes)

        self._dtype, self._shape = array.dtype, array.shape

        # Create Numpy array which uses the Shared Memory buffer, to copy in the input array.
        res = numpy.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared_buffer.buf
        )

        # Copy the input array into Shared Memory
        res[:] = array[:]

    def read(self):
        '''
        Reads the Shared Memory Segment and interprets it as a numpy array, with the underlying buffer.
        '''
        return numpy.ndarray(self._shape, self._dtype, buffer=self._shared_buffer.buf)

    def copy(self):
        '''
        Returns a copy of the numpy array.
        '''
        return numpy.copy(self.read())

    def unlink(self):
        '''
        Releases the underlying Shared Memory Segment, which holds the array.
        '''
        self._shared_buffer.close()
        self._shared_buffer.unlink()


class SharedDataFrame:
    '''
    Holds a Shared Dataframe.
    '''

    def __init__(self, dataframe):
        '''
        Create a Shared Memory Version of the Dataframe.
        '''
        self._values = SharedNumpyArray(dataframe.values)
        self._index = dataframe.index
        self._columns = dataframe.columns

    def read(self):
        '''
        Reads the Shared Memory Segment and interprets it as a pandas dataframe, with the underlying shared numpy arrays.
        '''
        return pandas.DataFrame(
            self._values.read(),
            index=self._index,
            columns=self._columns
        )

    def copy(self):
        '''
        Returns a copy of the Shared Dataframe
        '''
        return pandas.DataFrame(
            self._values.copy(),
            index=self._index,
            columns=self._columns
        )

    def unlink(self):
        '''
        Releases the underlying buffers, which were used to create the Shared Dataframe
        '''
        self._values.unlink()

########################################
