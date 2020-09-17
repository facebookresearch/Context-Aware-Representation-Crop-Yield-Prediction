#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from netCDF4 import Dataset
import numpy as np


def get_origi_lat_lon():
    in_dir = '../../processed_data/prism/monthly'
    lats, lons = None, None

    for f in os.listdir(in_dir):
        if f.endswith('.nc'):
            fh = Dataset(os.path.join(in_dir, f), 'r')
            if lats is None and lons is None:
                lats, lons = fh.variables['lat'][:], fh.variables['lon'][:]
            else:
                assert np.allclose(lats, fh.variables['lat'][:])
                assert np.allclose(lons, fh.variables['lon'][:])

    out_dir = '../../processed_data/prism/latlon'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(os.path.join(out_dir, 'lat_4km.npy'), lats.compressed())
    np.save(os.path.join(out_dir, 'lon_4km.npy'), lons.compressed())


def get_lat_lon_even(n=10):
    """
    :param n: how many pixels in lat or lon constructs one cell, e.g. n = 10 means the cell will be ~40 km * 40 km
    """
    origi_lats = np.load('../../processed_data/prism/latlon/lat_4km.npy')
    origi_lons = np.load('../../processed_data/prism/latlon/lon_4km.npy')
    # print('Lengths of origi: ', len(origi_lats), len(origi_lons))

    n_cell_lat = len(origi_lats)//n
    n_cell_lon = len(origi_lons)//n

    new_lats = []
    new_lons = []

    for i in range(n_cell_lat):
        i1, i2 = (n//2-1) + n * i, n//2 + n * i
        new_lats.append((origi_lats[i1] + origi_lats[i2])/2)

    for i in range(n_cell_lon):
        i1, i2 = (n // 2 - 1) + n * i, n // 2 + n * i
        new_lons.append((origi_lons[i1] + origi_lons[i2])/2)

    out_dir = '../../processed_data/prism/latlon'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(os.path.join(out_dir, 'lat_{}km.npy'.format(4*n)), np.asarray(new_lats))
    np.save(os.path.join(out_dir, 'lon_{}km.npy').format(4*n), np.asarray(new_lons))


if __name__ == "__main__":
    # get_origi_lat_lon()
    get_lat_lon_even()
