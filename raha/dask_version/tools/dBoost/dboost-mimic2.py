#! /usr/bin/env python3
import sqlite3
import sys
from utils import db
from utils.printing import print_rows
import dboost
import features
import cli

TABLES = [line.strip() for line in """
db_schema
d_chartitems_detail
d_parammap_items
d_careunits
d_demographicitems
d_meditems
d_patients
d_labitems
comorbidity_scores
drgevents
admissions
demographic_detail
icustay_detail
icustayevents
censusevents
d_codeditems
d_chartitems
deliveries
additives
procedureevents
d_ioitems
demographicevents
icustay_days
a_meddurations
d_caregivers
icd9
a_iodurations
microbiologyevents
parameter_mapping
noteevents
poe_order
totalbalevents
a_chartdurations
medevents
poe_med
ioevents
labevents
chartevents
""".splitlines() if line != ""]

COUNT = "SELECT COUNT(*) FROM {}"
QUERY = "SELECT * FROM {}"
#PATH = "/afs/csail.mit.edu/group/db/6830/mimic2.db"

parser = cli.get_mimic_parser()
args, models, rules = cli.parsewith(parser)

for table in TABLES:
    count_query = COUNT.format(table)
    row_count = db.read_db(args.path, count_query)[0][0]

    if row_count == 0:
        print("Skipping {} ({} rows)".format(table, row_count))
    else:
        print("Processing {} ({} rows)".format(table, row_count))

        query = QUERY.format(table)
        for model in models:
            model.reset()
            data = lambda: db.iter_db(args.path, query)
            outliers = list(dboost.outliers_streaming(data, model, rules))

            print("... {} found".format(len(outliers)))
            if 0 < len(outliers) < 200:
                print_rows(outliers, model, features.descriptions(rules), args.verbosity)

    print()

# Relevant strategies: 
# * Get a histogram of the data instead of assuming it's gaussian
#
# Extra features
# * sign(int)
