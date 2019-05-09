import dataset
import os

dataset_dictionary = {
        "name": "toy",
        "path": os.path.join("datasets", "toy", "dirty.csv"),
        "clean_path": os.path.join("datasets", "toy", "clean.csv")
    }

d = dataset.Dataset(dataset_dictionary)
