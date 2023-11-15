#! /usr/bin/env python3
import sys

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import dboost
from dboost import features
from dboost import cli
from dboost.utils.read import stream_tuples
from dboost.utils.printing import print_rows, debug

parser = cli.get_stdin_parser()
args, models, analyzers, rules = cli.parsewith(parser)

testset_generator = stream_tuples(args.input, args.fs, args.floats_only, args.inmemory, args.maxrecords)

if args.trainwith == None:
    args.trainwith = args.input
    trainset_generator = testset_generator
else:
    trainset_generator = stream_tuples(args.trainwith, args.fs, args.floats_only, args.inmemory, args.maxrecords)

if not args.inmemory and not args.trainwith.seekable():
    parser.error("Input does not support streaming. Try using --in-memory or loading input from a file?")

# TODO: Input should be fed to all models in one pass.
for model in models:
    for analyzer in analyzers:
        outliers = list(dboost.outliers(trainset_generator, testset_generator,
                                        analyzer, model, rules,args.runtime_progress, args.maxrecords))

        if len(outliers) == 0:
            debug("   All clean!")
        else:
            print_rows(outliers, model, analyzer.hints,
                       features.descriptions(rules), args.verbosity,dataset_name=args.input.name)
            debug("   {} outliers found".format(len(outliers)))