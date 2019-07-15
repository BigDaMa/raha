import dataset
import os

DATASETS_FOLDER = "datasets"
DATASET_NAME = "flights"

dataset_dictionary = {
        "name": DATASET_NAME,
        "path": os.path.join(DATASETS_FOLDER, DATASET_NAME, "dirty.csv"),
        "clean_path": os.path.join(DATASETS_FOLDER, DATASET_NAME, "clean.csv")
    }

d = dataset.Dataset(dataset_dictionary)

katara_data = d.dataframe.to_numpy().tolist()

sp_folder_path = ""

all_strategies = {}

cells_strategies = {}

ERROR_DETECTION_TOOLS = []

fv = {}

clusters_j_k_c_ce = {}

cells_clusters_j_k_ce = {}
