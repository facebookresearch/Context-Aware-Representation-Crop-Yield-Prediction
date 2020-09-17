#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import datetime
import os

import sys
sys.path.append("..")

from data_preprocessing.rescaling.rescale_utils import search_kdtree


def reproject_lat_lon():
    fh_lst = Dataset('../../raw_data/lst/1km/20170101.nc', 'r')
    lats, lons = fh_lst.variables['lat'][:], fh_lst.variables['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    fh_ref = Dataset('../../processed_data/lai/500m/20181028.nc', 'r')
    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]

    xv, yv = np.meshgrid(ref_lons, ref_lats)
    points = np.dstack([yv.ravel(), xv.ravel()])[0]
    print('Finish building points')
    results = search_kdtree(lats, lons, points)
    np.save('../../raw_data/lst/projected_indices_lai_500m.npy', results)


def reproject_lst(doy):
    print(doy)
    fh_ref = Dataset('../../processed_data/lai/500m/20181028.nc', 'r')
    fh_in = Dataset('../../raw_data/lst/1km/{}.nc'.format(doy), 'r')
    fh_out = Dataset('../../processed_data/lst/500m/{}.nc'.format(doy), 'w')

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
            new_name = '_'.join(v_name.lower().split('_')[:-1])
            outVar = fh_out.createVariable(new_name, varin.datatype, ('lat', 'lon'))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            origi_values[new_name] = varin[:]
            projected_values[new_name] = np.full((n_lat, n_lon), -9999.9)

    projected_indices = np.load('../../raw_data/lst/projected_indices_lai_500m.npy')
    projected_i = 0

    for i in range(n_lat):
        for j in range(n_lon):
            for key in origi_values.keys():
                proj_i, proj_j = projected_indices[projected_i] // 7797, projected_indices[projected_i] % 7797
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
    for doy in os.listdir('../../raw_data/lst/1km/'):
        if doy.endswith('.nc'):
            doy = doy[:-3]
            date = datetime.datetime.strptime(doy, "%Y%m%d").date()
            date_start = datetime.datetime.strptime('20190602', "%Y%m%d").date()
            date_end = datetime.datetime.strptime('20190602', "%Y%m%d").date()
            if date_start <= date <= date_end:
                reproject_lst(doy)
