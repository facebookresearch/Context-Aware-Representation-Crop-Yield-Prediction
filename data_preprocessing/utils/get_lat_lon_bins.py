#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def get_lat_lon_bins(lats, lons):
    inter_lat = np.array([(x + y) / 2.0 for x, y in zip(lats[:-1], lats[1:])])
    inter_lon = np.array([(x + y) / 2.0 for x, y in zip(lons[:-1], lons[1:])])
    lat_bins = np.concatenate([[2 * inter_lat[0] - inter_lat[1]], inter_lat, [2 * inter_lat[-1] - inter_lat[-2]]])
    lon_bins = np.concatenate([[2 * inter_lon[0] - inter_lon[1]], inter_lon, [2 * inter_lon[-1] - inter_lon[-2]]])

    return lats, lons, lat_bins, lon_bins