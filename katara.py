

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


def domain_spec_col_type(data, i, domain_specific_types, col_2_errors_repair):
    values = [row[i] for row in data]

    lowercase_types = [domain_type.lower() for domain_type in domain_specific_types]

    count = 0

    for value in values:
        if value.lower() in lowercase_types:
            count += 1

    coverage = count/len(values)

    if coverage > 0.2:
        for index, row in enumerate(data):
            value = values[index]
            if value.lower() not in lowercase_types and value not in domain_specific_types:
                fix = ""
                col_2_errors_repair[str(index) + "," + str(i)] = fix



def domain_spec_colpair(data, i, j, rel2sub2obj, col_2_errors_repair):
    for rel in rel2sub2obj:
        count = 0
        for row in data:
            coli = row[i]
            colj = row[j]
            if coli in rel2sub2obj[rel] and colj == rel2sub2obj[rel][coli]:
                count += 1

        coverage = count/ data.dataframe.shape[0]

        if coverage >= 0.15:
            for index, row in enumerate(data):
                coli = row[i]
                colj = row[j]
                if coli not in rel2sub2obj[rel] or colj != rel2sub2obj[rel][coli]:
                    if coli in rel2sub2obj[rel]:
                        repair_value = rel2sub2obj[rel][coli]
                        col_2_errors_repair[str(index) + "," + str(j)] = repair_value



def run_katara(data, domin_specific_file):
    domain_specific_types = []
    rel2sub2obj = {}
    col_2_errors_repair = {}

    load_file(domin_specific_file, domain_specific_types, rel2sub2obj)

    for i in range(len(data[0])):
        domain_spec_col_type(data, i, domain_specific_types, col_2_errors_repair)

    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            domain_spec_colpair(data, i, j, rel2sub2obj, col_2_errors_repair)

    return col_2_errors_repair
