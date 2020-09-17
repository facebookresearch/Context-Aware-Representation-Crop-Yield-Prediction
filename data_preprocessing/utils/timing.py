#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
import atexit
from time import time, clock
from time import strftime, localtime
import functools


def _secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        functools.reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60])


def _log(s, elapsed=None):
    line = "=" * 40
    print(line)
    print(s)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()


def _endlog(start):
    end = time()
    elapsed = end-start
    _log("End Program", _secondsToStr(elapsed))


def timenow():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()), _secondsToStr(clock()))


def timeit():
    start = time()
    atexit.register(_endlog, start)
    _log("Start Program")
