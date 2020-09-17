#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import date, timedelta


def generate_doy(s_doy, e_doy, delimiter):
    s_doy = map(int, [s_doy[:4], s_doy[4:6], s_doy[6:]])
    e_doy = map(int, [e_doy[:4], e_doy[4:6], e_doy[6:]])

    d1 = date(*s_doy)
    d2 = date(*e_doy)
    delta = d2 - d1

    for i in range(delta.days + 1):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_doy_every_n(s_doy, e_doy, n, delimiter):
    s_doy = map(int, [s_doy[:4], s_doy[4:6], s_doy[6:]])
    e_doy = map(int, [e_doy[:4], e_doy[4:6], e_doy[6:]])

    d1 = date(*s_doy)
    d2 = date(*e_doy)
    delta = d2 - d1

    for i in range(0, delta.days + 1, n):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_nearest_doys(doy, n, delimiter):
    doy = map(int, [doy[:4], doy[4:6], doy[6:]])
    d1 = date(*doy)

    for i in range((n+1)//2-n, (n+1)//2):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_most_recent_doys(doy, n, delimiter):
    doy = map(int, [doy[:4], doy[4:6], doy[6:]])
    d1 = date(*doy)

    for i in range(-1, -n-1, -1):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_future_doys(doy, n, delimiter):
    doy = map(int, [doy[:4], doy[4:6], doy[6:]])
    d1 = date(*doy)

    for i in range(n):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)
