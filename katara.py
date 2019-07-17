

def load_file(domin_specific_file, domain_specific_types, rel2sub2obj):
    for line in domin_specific_file:
        splits = line.replace('\n', '').split('\t')
        if len(splits) < 3:
            continue

        s = splits[0]
        rel = splits[1]
        o = splits[2]

        domain_specific_types.add(s)

        if rel in rel2sub2obj:
            rel2sub2obj[rel][s] = o
        else:
            rel2sub2obj[rel] = {s: o}


def domain_spec_col_type(data, col, domain_specific_types, col_2_errors_repair):
    values = [row[col] for row in data]

    lowercase_types = {domain_type.lower() for domain_type in domain_specific_types}

    count = 0
    tempdict = {}
    for index in range(len(data)):
        value = values[index]
        if value.lower() in lowercase_types and value.lower() != '':
            count += 1
        else:
            fix = ""
            tempdict[(index, col)] = fix

    coverage = count/len(values)

    if coverage > 0.2:
        col_2_errors_repair.update(tempdict)


def domain_spec_colpair(data, i, j, rel2sub2obj, col_2_errors_repair):
    for rel in rel2sub2obj:
        count = 0               # counts i to j relation
        back_count = 0          # counts j to i relation
        tempdict = {}
        backdict = {}
        for index, row in enumerate(data):
            coli = row[i]
            colj = row[j]
            if coli in rel2sub2obj[rel]:
                if coli != '' and colj == rel2sub2obj[rel][coli]:
                    count += 1
                else:
                    repair_value = rel2sub2obj[rel][coli]
                    tempdict[(index, j)] = repair_value
            if colj in rel2sub2obj[rel]:
                if colj != '' and coli == rel2sub2obj[rel][colj]:
                    back_count += 1
                else:
                    repair_value = rel2sub2obj[rel][colj]
                    backdict[(index, i)] = repair_value

        coverage = count/ len(data)
        backcoverage = back_count / len(data)

        if coverage >= 0.15:
            col_2_errors_repair.update(tempdict)         # adds the errors found to the final output
        if backcoverage >= 0.15:
            col_2_errors_repair.update(backdict)

def run_katara(data, domin_specific_file):
    domain_specific_types = set()
    rel2sub2obj = {}
    col_2_errors_repair = {}

    load_file(domin_specific_file, domain_specific_types, rel2sub2obj)

    for col in range(len(data[0])):
        domain_spec_col_type(data, col, domain_specific_types, col_2_errors_repair)

    for i in range(len(data[0])):
        for j in range(len(data[0])-1, i, -1):
            domain_spec_colpair(data, i, j, rel2sub2obj, col_2_errors_repair)

    return col_2_errors_repair
