#! /usr/bin/env python3
import sqlite3

def iter_db(path, query):
    with sqlite3.connect(path, detect_types = sqlite3.PARSE_COLNAMES) as connection:
        for row in connection.cursor().execute(query):
            yield tuple(row)

def read_db(path, query):
    return list(iter_db(path, query))
