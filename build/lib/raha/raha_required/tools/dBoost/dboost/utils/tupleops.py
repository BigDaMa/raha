from math import sqrt
from itertools import chain, combinations, product

def pair_ids(X, mask):
    for idx, idy in combinations(range(len(X)), 2):
        x, y = X[idx], X[idy]
        for idxi, idyi in product(range(len(x)), range(len(y))):
            if mask[idx][idxi] and mask[idy][idyi]:
                yield (idx, idxi), (idy, idyi)

def subtuple_ids(X, fundep_size):
    for ids in combinations(range(len(X)), fundep_size):
        for subids in product(*(range(len(X[idx])) for idx in ids)):
            yield tuple(zip(ids, subids))

def defaultif(S, X, default):
    return S if S != None else tuple(tuple(default() for _ in x) for x in X)

def defaultif_masked(S, X, default, mask):
    return S if S != None else tuple(tuple(default() if mi else None for (_, mi) in zip(x, m)) for x, m in zip(X, mask))

def zeroif(S, X):
    return S if S != None else tuple(tuple(0 for _ in x) for x in X)

def make_mask_abc(X, abc):
    return deepmap(lambda xi: isinstance(xi, abc), X)

def extract_types(X):
    return deepmap(type, X)

def types_consistent(ref, X):
    return extract_types(X) == ref

def compare_types(T1, T2):
    for i, (t1, t2) in enumerate(zip(T1, T2)):
        if t1 != t2:
            yield i

def addlist(S, n, d):
    if S is None: S = []
    if len(S) > n: return S
    S.append(list(0 for _ in range(d)))
    assert(len(S) >= n)
    return S

def addlist2d(S, n, d1, d2):
    if S is None: S = []
    if len(S) > n: return S
    S.append(list(list(0 for _ in range(d2)) for d in range(d1)))
    assert(len(S) >= n)
    return S

def root(X):
    return deepmap(sqrt, X) #TODO remove

def deepmap(f, X):
    return tuple(tuple(f(xi) for xi in x) for x in X)

def filter(f, X):
    return tuple(tuple((xi if (xi != None and f(xi)) else None) for xi in x) for x in X)

def filter_mask(X, mask):
    return tuple(tuple((xi if mi else None) for xi, mi in zip(x, m)) for x, m in zip(X, mask))

def merge(S, X, f, phi):
    return tuple(tuple(phi(si, f(xi)) for si, xi in zip(s, x)) for s, x in zip(S, X))

def deepapply(S, X, f):
    for s, x in zip(S, X):
        for si, xi in zip(s, x):
            f(si, xi)

def deepapply_masked(S, X, f, mask):
    for s, x, m in zip(S, X, mask):
        for si, xi, mi in zip(s, x, m):
            if mi:
                f(si, xi)

def number(X):
    return tuple(tuple((i, j) for j, _ in enumerate(x)) for i, x in enumerate(X))

def id(x):
    return x

def sqr(x):
    return x * x if x != None else None

def not_null(x):
    return x != None

def keep_if(a, b):
    return a if b else None

def plus(a, b):
    return a + b if b != None else a

def minus(a, b):
    return a - b if b != None else a

def mul(a, b):
    return a * b if b != None else a

def div0(a, b):
    return a / b if a != None and b != 0 else 0

def incrkey(a, b):
    if a != None:
        a[b] += 1
    return a

def tuplify(a, b):
    return (a, b)

def flatten(tup):
    return list(chain(*tup))

def filter_abc(X, abc):
    return tuple(tuple(xi for xi in x if isinstance(xi, abc)) for x in X)
