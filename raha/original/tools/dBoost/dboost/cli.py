import argparse
from . import features
from . import analyzers, models

REGISTERED_MODELS = models.ALL()
REGISTERED_ANALYZERS = analyzers.ALL()

def register_modules(parser, modules):
    for module in modules:
        module.register(parser)

def get_base_parser():
    base_parser = argparse.ArgumentParser(add_help = False)
    base_parser.add_argument("-v", "--verbose", dest = "verbosity",
                             action = "store_const", const = 1, default = 0,
                             help = "Print basic debugging information.")

    base_parser.add_argument("-vv", "--debug", dest = "verbosity",
                             action = "store_const", const = 2,
                             help = "Print advanced debugging information.")

    base_parser.add_argument("-d", "--disable-rule", dest = "disabled_rules",
                             action = "append", metavar = 'rule',
                             help = "Disable a tuple expansion rule.")

    base_parser.add_argument("--floats-only", dest = "floats_only",
                             action = "store_const", const = True, default = False,
                             help = "Parse all numerical fields as floats.")

    base_parser.add_argument("--max-records", dest = "maxrecords", metavar = "N",
                             action = "store", default = float("+inf"), type = int,
                             help = "Stop processing after reading at most N records.")

    base_parser.add_argument("--minimal", dest = "verbosity",
                             action = "store_const", const = -1,
                             help = "Trim output down to the bare minimum, reporting only the indices of outliers on stdout.")

    base_parser.add_argument("--pr", dest = "runtime_progress", metavar = "N",
                             action = "store", default = float("+inf"), type = int,
                             help = "Print runtime progress for every provided number of test set entries")
    base_parser.set_defaults(disabled_rules = [])

    register_modules(base_parser, REGISTERED_MODELS)
    register_modules(base_parser, REGISTERED_ANALYZERS)

    return base_parser

def get_stdin_parser():
    parser = argparse.ArgumentParser(parents = [get_base_parser()],
                                     description="Loads a database from a text file, and reports outliers")
    parser.add_argument("input", nargs='?', default = "-", type = argparse.FileType('r'),
                        help = "Read data from file input. If omitted or '-', read from standard input. Separate training data can be specified using --train-with")

    parser.add_argument("--train-with", dest = "trainwith", metavar = "input",
                        action = "store", default = None, type = argparse.FileType('r'),
                        help = "Use a separate dataset for correlation detection and model training. ")

    parser.add_argument("-m", "--in-memory",  dest = "inmemory",
                        action = "store_const", const = True, default = False,
                        help = "Load the entire dataset in memory before running. Required if input does not come from a seekable file.")

    parser.add_argument("-F", "--field-separator", dest = "fs",
                        action = "store", default = "\t", metavar = "fs",
                        help = "Use fs as the input field separator (default: tab).")

    return parser

def get_mimic_parser():
    parser = argparse.ArgumentParser(parents = [get_base_parser()],
                                     description="Loads the mimic2 database using sqlite3, and reports outliers")
    parser.add_argument("db", help = "Read data from sqlite3 database file db.")
    return parser

def load_modules(namespace, parser, registered_modules):
    modules = []

    for module in registered_modules:
        params = getattr(namespace, module.ID)
        if params != None:
            modules.append(module.from_parse(params))

    if len(modules) == 0:
        args = ["'--" + module.ID + "'" for module in registered_modules]
        parser.error("Please specify one of [{}]".format(", ".join(args)))

    return modules


def imported_parsewith(parser, args):
    args = parser.parse_args(args)
    models = load_modules(args, parser, REGISTERED_MODELS)
    analyzers = load_modules(args, parser, REGISTERED_ANALYZERS)

    disabled_rules = set(args.disabled_rules)
    available_rules = set(r.__name__ for rs in features.rules.values() for r in rs)
    invalid_rules = disabled_rules - available_rules
    if len(invalid_rules) > 0:
        parser.error("Unknown rule(s) {}. Known rules: {}".format(
            ", ".join(sorted(invalid_rules)),
            ", ".join(sorted(available_rules - disabled_rules))))
    rules = {t: [r for r in rs if r.__name__ not in disabled_rules]
             for t, rs in features.rules.items()}

    return args, models, analyzers, rules


def parsewith(parser):
    args = parser.parse_args()

    models = load_modules(args, parser, REGISTERED_MODELS)
    analyzers = load_modules(args, parser, REGISTERED_ANALYZERS)

    disabled_rules = set(args.disabled_rules)
    available_rules = set(r.__name__ for rs in features.rules.values() for r in rs)
    invalid_rules = disabled_rules - available_rules
    if len(invalid_rules) > 0:
        parser.error("Unknown rule(s) {}. Known rules: {}".format(
            ", ".join(sorted(invalid_rules)),
            ", ".join(sorted(available_rules - disabled_rules))))
    rules = {t: [r for r in rs if r.__name__ not in disabled_rules]
             for t, rs in features.rules.items()}

    return args, models, analyzers, rules

