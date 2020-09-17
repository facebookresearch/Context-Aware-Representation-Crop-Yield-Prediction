#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from scipy.spatial import cKDTree
import numpy as np


def search_kdtree(lats, lons, points):
    mytree = cKDTree(np.dstack([lats.ravel(), lons.ravel()])[0])
    print('Finish building KDTree')
    dist, indices = mytree.query(points)
    return indices


def get_lat_lon_bins(lats, lons):
    inter_lat = np.array([(x + y) / 2.0 for x, y in zip(lats[:-1], lats[1:])])
    inter_lon = np.array([(x + y) / 2.0 for x, y in zip(lons[:-1], lons[1:])])
    lat_bins = np.concatenate([[2 * inter_lat[0] - inter_lat[1]], inter_lat, [2 * inter_lat[-1] - inter_lat[-2]]])
    lon_bins = np.concatenate([[2 * inter_lon[0] - inter_lon[1]], inter_lon, [2 * inter_lon[-1] - inter_lon[-2]]])

    return lat_bins, lon_bins
