from collections import defaultdict
import inspect
import time
from .. import utils
import sys
import unicodedata
import re
import email.utils

# Rules are functions that take a value (field), and return a tuple of features
# derived from that value.

rules = defaultdict(list)

def rule(rule):
    spec = inspect.getfullargspec(rule)

    if len(spec.args) != 1:
        sys.stderr.write("Invalid rule {}".format(rule.__name__))
        sys.exit(1)

    input_type = spec.annotations[spec.args[0]]
    rules[input_type].append(rule)
    return rule

def descriptions(ruleset):
    descriptions = {}

    for type in ruleset:
        descriptions[type] = []
        for rule in ruleset[type]:
            descriptions[type].extend(inspect.getfullargspec(rule).annotations['return'])

    return descriptions

@rule
def string_case(s: str) -> ("upper case", "lower case", "title case"):
    return (s.isupper(), s.islower(), s.istitle())

@rule
def string_is_digit(s: str) -> ("is digit",):
    return (s.isdigit(),)

@rule
def length(s: str) -> ("length",):
    return (len(s),)

@rule
def signature(s: str) -> ("signature",):
    return (",".join(map(unicodedata.category, s)),)

NUMBERS = re.compile(r"(^s)?\d+")

@rule
def strp(s: str) -> ("strp",):
    return (NUMBERS.sub("<num>", s),)

HTML5_EMAIL_VALIDATOR = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.(?P<ext>[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*$")

@rule
def email_checks(s: str) -> ("simple email check",): # "RFC822 email check"):
    return (HTML5_EMAIL_VALIDATOR.match(s) != None,) #, (email.utils.parseaddr(s) != ('', ''))
#TODO: This should be asymmetric

@rule
def email_domain(s: str) -> ("email domain",):
    match = HTML5_EMAIL_VALIDATOR.match(s)
    return (match.group("ext").lower() if match and match.group("ext") else "NONE",)

@rule
def id(s: str) -> ("id",):
    return (s,)

@rule
def empty(s: str) -> ("empty",):
    return (s == "" or s.isspace(),)

@rule
def int_id(x: int) -> ("id",):
    return (x,)

@rule
def int_kill(x: int) -> ("nil",):
    return (None,)

#@rule
#def float_length(f: float) -> ("length",):
#  return(len(str(f)),)

@rule
def float_id(f: float) -> ("id",):
    return (f,)

# TODO add rule to parse dates from strings

def _bits(*positions):
    def bits(i: int) -> tuple("bit {}".format(pos) for pos in positions):
        return ((i >> pos) & 1 for pos in positions)
    return bits

def _mod(*mods):
    def mod(i: int) -> tuple("mod {}".format(mod) for mod in mods):
        return (i % mod for mod in mods)
    return mod

def _div(*mods):
    def div(i: int) -> tuple("div {}".format(mod) for mod in mods):
        return (i % mod == 0 for mod in mods)
    return div

DATE_PROPS = "tm_year", "tm_mon", "tm_mday", "tm_hour", "tm_min", "tm_sec", "tm_wday", "tm_yday"

@rule
def unix2date(timestamp: int) -> DATE_PROPS:
    t = time.gmtime(timestamp)
    return map(lambda a: getattr(t, a), DATE_PROPS)

@rule
def unix2date_float(timestamp: float) -> DATE_PROPS:
    return unix2date(int(timestamp))

@rule
def fracpart(x: float) -> ("frac part",):
    return (x - int(x),)

@rule
def is_weekend(timestamp: int) -> ("is weekend",):
    wday = time.gmtime(timestamp).tm_wday
    wkend = wday in [5, 6]
    return (wkend,)

rule(_bits(0, 1, 2, 3, 4, 5))
rule(_div(3, 5))
rule(_mod(10))
