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
import sys
import itertools
import pandas
########################################


########################################
class Dataset:
    """
    The dataset class.
    """

    def __init__(self, dataset_dictionary):
        """
        The constructor creates a dataset.
        """
        self.name = dataset_dictionary["name"]
        self.dataframe = self.read_csv_dataset(dataset_dictionary["path"])
        if "clean_path" in dataset_dictionary:
            self.clean_dataframe = self.read_csv_dataset(dataset_dictionary["clean_path"])
            if self.dataframe.shape != self.clean_dataframe.shape:
                sys.stderr.write("Ground truth is not in the equal size to the dataset!\n")
            self.actual_errors_dictionary = self.get_actual_errors_dictionary()
        if "repaired_path" in dataset_dictionary:
            self.repaired_dataframe = self.read_csv_dataset(dataset_dictionary["repaired_path"])
            if self.dataframe.shape != self.repaired_dataframe.shape:
                sys.stderr.write("Repaired dataset is not in the equal size to the dataset!\n")
            self.repairs_dictionary = self.get_repairs_dictionary()

    def read_csv_dataset(self, dataset_path):
        """
        This method reads a dataset from a csv file path.
        """
        dataset_dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                            keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
        return dataset_dataframe

    def write_csv_dataset(self, dataset_path, dataframe):
        """
        This method writes a dataset to a csv file path.
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

    def get_actual_errors_dictionary(self):
        """
        This method compares the clean and dirty versions of a dataset.
        """
        return {(i, j): self.clean_dataframe.iloc[i, j]
                for (i, j) in itertools.product(range(self.dataframe.shape[0]), range(self.dataframe.shape[1]))
                if self.dataframe.iloc[i, j] != self.clean_dataframe.iloc[i, j]}

    def create_repaired_dataset(self, correction_dictionary):
        """
        This method takes the dictionary of corrected values and creates the repaired dataset.
        """
        self.repaired_dataframe = self.dataframe.copy()
        for cell in correction_dictionary:
            self.repaired_dataframe.iloc[cell] = correction_dictionary[cell]

    def get_repairs_dictionary(self):
        """
        This method compares the repaired and dirty versions of a dataset.
        """
        return {(i, j): self.repaired_dataframe.iloc[i, j]
                for (i, j) in itertools.product(range(self.dataframe.shape[0]), range(self.dataframe.shape[1]))
                if self.dataframe.iloc[i, j] != self.repaired_dataframe.iloc[i, j]}

    def get_data_quality(self):
        """
        This method calculates data quality of a dataset.
        """
        return 1.0 - float(len(self.actual_errors_dictionary)) / (self.dataframe.shape[0] * self.dataframe.shape[1])

    def evaluate_data_cleaning(self, correction_dictionary, sampled_rows_dictionary=False):
        """
        This method evaluates data cleaning process.
        """
        actual_errors = dict(self.actual_errors_dictionary)
        if sampled_rows_dictionary:
            actual_errors = {(i, j): self.actual_errors_dictionary[(i, j)]
                             for (i, j) in self.actual_errors_dictionary if i in sampled_rows_dictionary}
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
########################################


########################################
if __name__ == "__main__":

    dataset_dictionary = {
        "name": "toy",
        "path": "datasets/dirty.csv",
        "clean_path": "datasets/clean.csv"
    }
    d = Dataset(dataset_dictionary)
    print d.get_data_quality()
########################################
