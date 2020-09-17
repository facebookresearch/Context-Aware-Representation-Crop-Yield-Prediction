#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import fiona

import sys
sys.path.append("..")

from data_preprocessing.utils import generate_doy
from data_preprocessing.preprocess import search_kdtree


def extract_shapefile():
    shapefile = fiona.open('../../raw_data/nws_precip/nws_precip_allpoint_conversion/nws_precip_allpoint_conversion.shp')

    lats = np.full((881, 1121), np.inf)
    lons = np.full((881, 1121), np.inf)
    max_hrapx, max_hrapy = -float('inf'), -float('inf')
    for feature in shapefile:
        hrapx, hrapy = feature['properties']['Hrapx'], feature['properties']['Hrapy']
        max_hrapx = max(max_hrapx, hrapx)
        max_hrapy = max(max_hrapy, hrapy)
        lon, lat = feature['geometry']['coordinates']
        if 0 <= hrapx < 1121 and 0 <= hrapy < 881:
            lats[hrapy, hrapx] = lat
            lons[hrapy, hrapx] = lon
    print(max_hrapx, max_hrapy)
    np.save('../../raw_data/nws_precip/nws_precip_allpoint_conversion/lats.npy', lats)
    np.save('../../raw_data/nws_precip/nws_precip_allpoint_conversion/lons.npy', lons)


def compute_closest_grid_point(lats, lons, lat, lon):
    d_lats = lats - float(lat)
    d_lons = lons - float(lon)
    d = np.multiply(d_lats, d_lats) + np.multiply(d_lons, d_lons)
    i, j = np.unravel_index(d.argmin(), d.shape)
    return i, j, np.sqrt(d.min())


def reproject_lat_lon():
    lats = np.load('../../raw_data/nws_precip/nws_precip_allpoint_conversion/lats.npy')
    lons = np.load('../../raw_data/nws_precip/nws_precip_allpoint_conversion/lons.npy')

    fh_ref = Dataset('../../processed_data/lai/500m/20181028.nc', 'r')
    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]

    xv, yv = np.meshgrid(ref_lons, ref_lats)
    points = np.dstack([yv.ravel(), xv.ravel()])[0]
    print('Finish building points')
    results = search_kdtree(lats, lons, points)
    np.save('../../raw_data/nws_precip/nws_precip_allpoint_conversion/projected_indices_lai_500m.npy', results)


def reproject_nws_precip(doy):
    print(doy)
    fh_ref = Dataset('../../processed_data/lai/500m/20181028.nc', 'r')
    fh_in = Dataset('../../raw_data/nws_precip/{}/nws_precip_1day_{}_conus.nc'.format(doy, doy), 'r')
    fh_out = Dataset('../../processed_data/nws_precip/500m/{}.nc'.format(doy), 'w')

    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]
    n_lat, n_lon = len(ref_lats), len(ref_lons)
    for name, dim in fh_ref.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_ref.variables.items():
        if v_name in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    observed_values = fh_in.variables['observation'][:]
    projected_values = np.full((n_lat, n_lon), -9999.9)
    projected_indices = \
        np.load('../../raw_data/nws_precip/nws_precip_allpoint_conversion/projected_indices_lai_500m.npy')
    projected_i = 0
    for i in range(n_lat):
        for j in range(n_lon):
            proj_i, proj_j = 881 - projected_indices[projected_i] // 1121, projected_indices[projected_i] % 1121
            if not observed_values.mask[proj_i, proj_j]:
                projected_values[i, j] = observed_values[proj_i, proj_j]
            projected_i += 1

    outVar = fh_out.createVariable('precip', 'f4', ('lat', 'lon'))
    outVar[:] = ma.masked_equal(projected_values, -9999.9)

    fh_in.close()
    fh_ref.close()
    fh_out.close()


if __name__ == '__main__':
    # extract_shapefile()
    # reproject_lat_lon()
    for doy in generate_doy('20171227', '20171231', ''):
        reproject_nws_precip(doy)
