#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def match_lat_lon(lats_from, lons_from, lats_to, lons_to, expand=0):
    i_lat_start = i_lat_end = i_lon_start = i_lon_end = 0

    for i in range(len(lats_from)):
        if abs(lats_from[i] - lats_to[0]) < 0.00001:
            i_lat_start = i - expand
        if abs(lats_from[i] - lats_to[-1]) < 0.00001:
            i_lat_end = i + expand
    for i in range(len(lons_from)):
        if abs(lons_from[i] - lons_to[0]) < 0.00001:
            i_lon_start = i - expand
        if abs(lons_from[i] - lons_to[-1]) < 0.00001:
            i_lon_end = i + expand

    return i_lat_start, i_lat_end, i_lon_start, i_lon_end
