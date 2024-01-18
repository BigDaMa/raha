import sys
from abc import ABC, abstractmethod
import html
import re

import pandas


class Dataset(ABC):

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

    @staticmethod
    def read_csv_dataset(dataset_path):
        """
        This method reads a dataset from a csv file path.
        """
        dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(Dataset.value_normalizer)
        return dataframe

    @staticmethod
    def write_csv_dataset(dataset_path, dataframe):
        """
        This method writes a dataset to a csv file path.
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

    @abstractmethod
    def get_data_quality(self):
        """
        This method calculates data quality of a dataset.
        """
        pass

    @abstractmethod
    def get_correction_dictionary(self):
        """
        This method compares the repaired and dirty versions of a dataset.
        """
        pass

    @abstractmethod
    def get_actual_errors_dictionary(self):
        """
        This method compares the clean and dirty versions of a dataset.
        """
        pass

    @abstractmethod
    def create_repaired_dataset(self, correction_dictionary):
        """
        This method takes the dictionary of corrected values and creates the repaired dataset.
        """
        pass
