#!/usr/bin/env python3
# Generates a fizzbuzz, with some mistakes

import utils

MAX = 1000
OUTPUT = "fizzbuzz"
FORMAT = "{}\t{}\n"

def writeout(output, num, msg):
    output.write(FORMAT.format(num, msg))

with open(utils.abspath(OUTPUT), mode = "w") as output:
    for num in range(MAX + 1):
        three = (num % 3) == 0
        five = (num % 5) == 0

        if num == 28:
            writeout(output, num, "Woof!")
            continue
        if num == 25:
            writeout(output, num, "Fizz")
            continue
        if num == 30:
            writeout(output, num, "Buzz")
            continue

        if three and five:
            writeout(output, num, "FizzBuzz")
        elif three:
            writeout(output, num, "Fizz")
        elif five:
            writeout(output, num, "Buzz")
        else:
            writeout(output, num, "s" + "{:0>5}".format(num))
