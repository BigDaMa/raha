from .autoconv import autoconv
from .printing import debug
import sys, csv

def parse_line_blind(row, floats_only):
    return tuple(autoconv(field, floats_only) for field in row)

def stream_tuples(input, fs, floats_only, preload, maxrecords = float("+inf")):
    def stream():
        if stream.call_count > 0:
            input.seek(0)
        stream.call_count += 1

        for rid, row in enumerate(csv.reader(input, delimiter = fs)):
            if rid > maxrecords:
                break

            if stream.row_length == None:
                stream.row_length = len(row)
            elif len(row) != stream.row_length:
                sys.stderr.write("Discarding {} (invalid length)\n".format(row))
                continue

            if stream.types == None:
                row = parse_line_blind(row, floats_only)
                stream.types = tuple(map(type, row))
            else:
                try:
                    row = tuple(conv(field) for conv, field in zip(stream.types, row))
                except ValueError:
                    sys.stderr.write("Discarding {} (invalid types)\n".format(row))
                    continue

            if stream.call_count == 1 and rid == 0 and stream.row_length == 1:
                debug("Your dataset seems to have only one column. Did you need -F?")

            yield row

    stream.call_count = 0
    stream.types = None
    stream.row_length = None

    if preload:
        dataset = list(stream())
        return (lambda: dataset)
    else:
        return stream
