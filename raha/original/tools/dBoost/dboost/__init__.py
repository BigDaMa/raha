#! /usr/bin/env python3
from .utils import tupleops
from .utils.printing import debug
from itertools import chain
import timeit,sys

def expand_field(f, rules):
    rls = rules[type(f)]
    return tuple(chain.from_iterable(rule(f) for rule in rls))

def expand(x, rules):
    return tuple(expand_field(f, rules) for f in x)

def expand_hints(X, hints):
    expanded_hints = tupleops.deepmap(lambda h: X[h[0]][h[1]], hints)
    return (expanded_hints,) + X

def expand_stream(generator, rules, keep_x, hints, maxrecords = float("+inf")):
    for idx, x in enumerate(generator()):
        if idx >= maxrecords:
            break
        X = expand(x, rules)
        if hints is not None:
            X = expand_hints(X, hints)
        yield (x, X) if keep_x else X

def outliers(trainset_generator, testset_generator, analyzer, model, rules,runtime_progress, maxrecords = float("+inf")):
    start = timeit.default_timer()
    debug(">> Finding correlations")

    analyzer.fit(expand_stream(trainset_generator, rules, False, None, maxrecords))
    # debug(analyzer.hints)

    debug(">> Building model...")
    analyzer.expand_stats()
    model.fit(expand_stream(trainset_generator, rules, False, analyzer.hints, maxrecords), analyzer)

    debug(">> Finding outliers...")
    for index, (x, X) in enumerate(expand_stream(testset_generator, rules,
                                                 True, analyzer.hints, maxrecords)):
        discrepancies = model.find_discrepancies(X, index)
        if len(discrepancies) > 0:
            yield index, (x, X, discrepancies)
        if index % runtime_progress == 0:
            debug("Time {} {}".format(index,timeit.default_timer()-start))
    stop = timeit.default_timer()
    debug("Runtime ",stop-start)
