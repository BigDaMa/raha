#! /usr/bin/env python3
def run(params):
    import sys

    if __name__ == '__main__' and __package__ is None:
        from os import sys, path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from . import features, cli, outliers
    from .utils.read import stream_tuples
    from .utils.printing import print_rows, debug

    parser = cli.get_stdin_parser()
    args, models, analyzers, rules = cli.imported_parsewith(parser, params)

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
            outlier_cells = list(outliers(trainset_generator, testset_generator, 
                                          analyzer, model, rules,args.runtime_progress, args.maxrecords))   # outliers is defined in __init__.py

            if len(outlier_cells) == 0:
                debug("   All clean!")
            else:
                print_rows(outlier_cells, model, analyzer.hints,
                           features.descriptions(rules), args.verbosity, dataset_name=args.input.name)
                debug("   {} outliers found".format(len(outlier_cells)))
