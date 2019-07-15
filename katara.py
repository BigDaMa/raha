

def load_file(domin_specific_file, domain_specific_types, rel2sub2obj):
    for line in domin_specific_file:
        splits = line.replace('\n', '').split('\t')
        if len(splits) < 3:
            continue

        s = splits[0]
        rel = splits[1]
        o = splits[2]

        domain_specific_types.append(s)

        if rel in rel2sub2obj:
            rel2sub2obj[rel][s] = o
        else:
            rel2sub2obj[rel] = {s: o}


def domain_spec_col_type(data, i, col, domain_specific_types, col_2_errors, domain_specific_covered_cols, col_2_errors_repair):
    values = data.dataframe[col]

    for type in domain_specific_types:
        type = type.lower()

        count = 0

        for value in values:
            if value.lower() == type:
                count += 1

        coverage = count/len(values)

        if coverage > 0.2:
            errorTuples = {}    #will be the indexes of the erronous tuples
            for index, row in data.dataframe.iterrows():
                value = row[col]
                if value.lower() != type:
                    errorTuples.add(index)
                    fix = ""
                    col_2_errors_repair[str(index) + "," + str(i)] = fix

            col_2_errors[col] = errorTuples
            domain_specific_covered_cols.add(col)
            break


def domain_spec_colpair(data, i, col_1, j, col_2, rel2sub2obj, col_2_errors, col_2_errors_repair):
    for rel in rel2sub2obj:
        count = 0
        for _, row in data.dataframe.iterrows():
            coli = row[col_1]
            colj = row[col_2]
            if coli in rel2sub2obj[rel] and colj == rel2sub2obj[rel][coli]:
                count += 1

        coverage = count/ data.dataframe.shape[0]

        if coverage >= 0.15:
            error_tuples = {}
            error_tuples_repair = {}

            for index, row in data.dataframe.iterrows():
                coli = row[col_1]
                colj = row[col_2]
                if coli not in rel2sub2obj[rel] or colj != rel2sub2obj[rel][coli]:
                    error_tuples.add(index)     # the index of the row

                    if coli in rel2sub2obj[rel]:
                        repair_value = rel2sub2obj[rel][coli]
                        col_2_errors_repair[str(index) + "," + str(j)] = repair_value

            col_2_errors[str(i) + "," + str(j)] = error_tuples


def run_katara(data, domin_specific_file):
    domain_specific_types = []
    rel2sub2obj = {}
    col_2_errors = {}
    col_2_errors_repair = {}
    domain_specific_covered_cols = {}

    load_file(domin_specific_file, domain_specific_types, rel2sub2obj)

    for index, column in enumerate(data.dataframe.columns):
        domain_spec_col_type(data, index, column, domain_specific_types, col_2_errors, domain_specific_covered_cols)

    for i, col_1 in enumerate(data.dataframe.columns):
        for j, col_2 in enumerate(data.dataframe.columns):
            if i == j:
                continue
            domain_spec_colpair(data, i, col_1, j, col_2, rel2sub2obj, col_2_errors, col_2_errors_repair)

    return col_2_errors_repair
