#! /usr/bin/env python3
class term:
    TEMPLATE = '[{};{}m'
    COLORLESS_TEMPLATE = '[{}m'
    
    PLAIN     = 0
    BOLD      = 1
    DIM       = 2
    UNDERLINE = 4
    BLACK     = 30
    RED       = 31
    GREEN     = 32
    YELLOW    = 33
    BLUE      = 34
    PURPLE    = 35
    CYAN      = 36
    MAGENTA   = 37
    WHITE     = 38
    RESET     = '[0;0m'

def highlight(msg, attr = term.BOLD, color = term.RED):
    return term.RESET + term.TEMPLATE.format(attr, color) + msg + term.RESET

def underline(msg):
    return term.RESET + term.COLORLESS_TEMPLATE.format(term.UNDERLINE) + msg + term.RESET
