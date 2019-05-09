import dataset
import os

DATASETS_FOLDER = "datasets"

dataset_dictionary = {
        "name": "toy",
        "path": os.path.join(DATASETS_FOLDER, "toy", "dirty.csv"),
        "clean_path": os.path.join(DATASETS_FOLDER, "toy", "clean.csv")
    }

d = dataset.Dataset(dataset_dictionary)
