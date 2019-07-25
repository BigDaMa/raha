import raha
import os

# --------------------
DATASET_NAME = "flights"
dataset_dictionary = {
    "name": DATASET_NAME,
    "path": os.path.join(os.path.dirname(__file__), "datasets", DATASET_NAME, "dirty.csv"),
    "clean_path": os.path.join(os.path.dirname(__file__), "datasets", DATASET_NAME, "clean.csv")
}

application = raha.Raha(dataset_dictionary)
# --------------------
application.run()
# --------------------
