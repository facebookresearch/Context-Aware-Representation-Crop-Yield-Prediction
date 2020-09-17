#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import pandas as pd
import csv

import sys
sys.path.append("..")

from data_preprocessing.rescaling.rescale_utils import search_kdtree


def reproject_lat_lon():
    fh_sf = Dataset('../../raw_data/soil_fraction/soil_fraction_usa.nc', 'r')
    lats, lons = fh_sf.variables['lat'][:], fh_sf.variables['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    fh_ref = Dataset('../../processed_data/lst/monthly_1km/201701.nc', 'r')
    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]

    xv, yv = np.meshgrid(ref_lons, ref_lats)
    points = np.dstack([yv.ravel(), xv.ravel()])[0]
    print('Finish building points')
    results = search_kdtree(lats, lons, points)
    np.save('../../raw_data/soil_fraction/projected_indices_lst_1km.npy', results)


def reproject_sf():
    fh_ref = Dataset('../../processed_data/lst/monthly_1km/201701.nc', 'r')
    fh_in = Dataset('../../raw_data/soil_fraction/soil_fraction_usa.nc', 'r')
    fh_out = Dataset('../../processed_data/soil_fraction/soil_fraction_usa_1km.nc', 'w')

    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]
    n_lat, n_lon = len(ref_lats), len(ref_lons)
    for name, dim in fh_ref.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_ref.variables.items():
        if v_name in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    origi_values = {}
    projected_values = {}
    for v_name, varin in fh_in.variables.items():
        if v_name not in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, 'f4', ('lat', 'lon'))
            outVar.setncatts({'_FillValue': np.array([-9999.9]).astype('f')})
            origi_values[v_name] = varin[:]
            projected_values[v_name] = np.full((n_lat, n_lon), -9999.9)

    projected_indices = np.load('../../raw_data/soil_fraction/projected_indices_lst_1km.npy')
    projected_i = 0
    for i in range(n_lat):
        for j in range(n_lon):
            for key in origi_values.keys():
                proj_i, proj_j = projected_indices[projected_i] // 8724, projected_indices[projected_i] % 8724
                if not origi_values[key].mask[proj_i, proj_j]:
                    projected_values[key][i, j] = origi_values[key][proj_i, proj_j]
            projected_i += 1

    for key in origi_values.keys():
        fh_out.variables[key][:] = ma.masked_equal(projected_values[key], -9999.9)

    fh_in.close()
    fh_ref.close()
    fh_out.close()


if __name__ == '__main__':
    # reproject_lat_lon()
    reproject_sf()
