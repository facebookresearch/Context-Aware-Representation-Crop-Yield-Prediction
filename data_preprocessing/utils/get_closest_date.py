#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import date


def get_closet_date(query_date, folder):
    doys = [x[:-3] for x in os.listdir(folder) if x.endswith('.nc')]
    doys = [date(*map(int, [x[:4], x[4:6], x[6:]])) for x in doys]
    query_date = date(*map(int, [query_date[:4], query_date[4:6], query_date[6:]]))

    return str(min(doys, key=lambda x: abs(x - query_date))).replace('-', '')


if __name__ == '__main__':
    print(get_closet_date('20170101', ['20161230', '20170503', '20170105']))

