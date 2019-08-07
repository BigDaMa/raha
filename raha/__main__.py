########################################
# Benchmarking Raha
# If you have a dirty dataset and its
# corresponding cleaned dataset and you
# just want to benchmark Raha, then you
# can use the following example.
########################################


########################################
import raha
import os
########################################


########################################
app = raha.Raha()
app.RUN_COUNT = 10   # Raha will be run RUN_COUNT times and, in the end, you will see the mean and std of the results.
app.LABELING_BUDGET = 20   # Raha will use up to LABELING_BUDGET labeled tuples from the clean dataset.
dataset_dictionary = {
    "name": "flights",
    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "flights", "dirty.csv")),
    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "flights", "clean.csv"))
}
app.run(dataset_dictionary)
########################################
