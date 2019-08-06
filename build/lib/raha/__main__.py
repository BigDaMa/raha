import raha
import os

# --------------------
DATASET_NAME = "hospital"
DATASET_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "datasets", DATASET_NAME)
dataset_dictionary = {
    "name": DATASET_NAME,
    "path": os.path.join(DATASET_PATH, "dirty.csv"),
    "clean_path": os.path.join(DATASET_PATH, "clean.csv")
}

application = raha.Raha(dataset_dictionary)
# --------------------
application.run()
# --------------------
