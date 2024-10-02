import os
import sys
import csv
import bisect
from . import color

def debug(*args, **kwargs):
    kwargs["file"] = sys.stderr
    #print(*args, **kwargs)

def report_progress(nb):
    if nb % 1000 == 0:
        print(nb, end="\r", file=sys.stderr)

def expand_hints(fields_group, hints):
    expanded_group = []

    for field_id, feature_id in fields_group:
        if field_id == 0:
            expanded_group.extend(hints[feature_id])
        else:
            expanded_group.append((field_id - 1, feature_id))

    return tuple(expanded_group)

def describe_discrepancy(fields_group, rules_descriptions, hints, x):
    expanded = expand_hints(fields_group, hints)

    field_ids, values, features = zip(*((field_id, x[field_id],
                                         rules_descriptions[type(x[field_id])][feature_id])
                                        for field_id, feature_id in expanded))

    if len(expanded) == 1:
        FMT = "   > Value '{}' ({}) doesn't match feature '{}'"
        msg = FMT.format(values[0], field_ids[0], features[0])
    else:
        FMT = "   > Values {} {} do not match features {}"
        msg = FMT.format(values, field_ids, features)

    return msg, features

def print_rows(outliers, model, hints, rules_descriptions, verbosity = 0, max_w = 40, header = "   ",dataset_name=""):
    if len(outliers) == 0:
        return

    # each outlier is (x, X, discrepancies)
    nb_fields = len(outliers[0][1][0])
    widths = (0,) * nb_fields

    # Compute the ideal column width for each column
    for _, (x, _, _) in outliers:
        widths = tuple(max(w, min(max_w, len(str(f))))
                       for w, f in zip(widths, x))

    results_file = open(dataset_name + "-dboost_output.csv", "w")
    csv_writer = csv.writer(results_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for linum, (x, X, discrepancies) in outliers:
        highlight = [field_id for fields_group in discrepancies
                              for field_id, _ in expand_hints(fields_group, hints)]
        column_value_dictionary = {}
        for fields_group in discrepancies:
            expanded = expand_hints(fields_group, hints)
            field_ids, values, features = zip(*((field_id, x[field_id],
                                                 rules_descriptions[type(x[field_id])][feature_id])
                                                for field_id, feature_id in expanded))
            for i in range(len(field_ids)):
                column_value_dictionary[field_ids[i]] = values[i]
        for column in column_value_dictionary:
            csv_writer.writerow([linum, column])
            # column_value_dictionary[column] is the detected value

def colorize(row, indices):
    row = [str(f) for f in row]
    for index in indices:
        row[index] = color.highlight(row[index], color.term.UNDERLINE)
    return row

def hhistplot(counter, highlighted, indent = "", pipe = sys.stdout, w = 20):
    BLOCK = "█"
    LEFT_HALF_BLOCK = "▌"

    try:
        W, H = os.get_terminal_size()
    except (OSError, AttributeError):
        W, H = 80, 24

    plot_w = min(w, W - 10 - len(indent))
    scale = plot_w / max(counter.values())
    data = sorted(counter.items())

    if highlighted  != None and highlighted not in counter:
        bisect.insort_left(data, (highlighted, 0))

    header_width = max(len(str(value)) for _, value in data)

    for key, value in data:
        label = str(key)
        bar_size = int(value * scale)
        header = indent + "[" + str(value).rjust(header_width) + "] "
        bar = (BLOCK * bar_size if bar_size > 0 else LEFT_HALF_BLOCK) + " "

        label_avail_space = W - 2 - len(bar) - len(header)
        if len(label) > label_avail_space:
            label = label[:label_avail_space - 3] + "..."

        line = bar + label
        if key == highlighted:
            line = color.highlight(line, color.term.PLAIN, color.term.RED)

        pipe.write(header + line + "\n")
