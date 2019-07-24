import utils
import os

def load_country_data():
    """Loads a list of [Country, Capital, Currency name, Currency symbol]"""
    lines = []
    with open(os.path.join(utils.BASE, "country_data")) as country_data:
        for line in country_data:
            line = line.strip().lower().split("\t")
            lines.append(tuple(line))
    return lines

COUNTRY_DATA = load_country_data()
