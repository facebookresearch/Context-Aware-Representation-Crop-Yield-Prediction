#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

import sys
sys.path.append("..")

from data_preprocessing.rescaling.rescale_utils import get_lat_lon_bins


def reproject_us_counties(in_file, ref_file, out_file):
    fh_in = Dataset(in_file, 'r')
    fh_out = Dataset(out_file, 'w')
    fh_ref = Dataset(ref_file, 'r')

    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]
    lat_bins, lon_bins = get_lat_lon_bins(ref_lats, ref_lons)

    origi_lats = fh_in.variables['lat']
    origi_lats_value = origi_lats[:]
    origi_lons = fh_in.variables['lon']
    origi_lons_value = origi_lons[:]
    origi_values = {}
    sampled_values = {}
    selected_vars = []
    for v in fh_in.variables:
        if v not in ['lat', 'lon']:
            selected_vars.append(v)
            origi_values[v] = fh_in.variables[v][:]
            sampled_values[v] = np.full((len(ref_lats), len(ref_lons)), 0)

    for id_lats in range(len(ref_lats)):
        for id_lons in range(len(ref_lons)):
            lats_index = np.searchsorted(origi_lats_value, [lat_bins[id_lats + 1], lat_bins[id_lats]])
            lons_index = np.searchsorted(origi_lons_value, [lon_bins[id_lons], lon_bins[id_lons + 1]])
            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                for v in selected_vars:
                    selected = origi_values[v][np.array(range(lats_index[0], lats_index[1])),
                                               np.array(range(lons_index[0], lons_index[1]))]
                    if selected.count() > 0:
                        sampled_values[v][id_lats, id_lons] = np.bincount(selected.compressed()).argmax()

                    else:
                        sampled_values[v][id_lats, id_lons] = 0
        print(id_lats)

    fh_out.createDimension('lat', len(ref_lats))
    fh_out.createDimension('lon', len(ref_lons))

    outVar = fh_out.createVariable('lat', 'f4', ('lat',))
    outVar.setncatts({k: origi_lats.getncattr(k) for k in origi_lats.ncattrs()})
    outVar[:] = ref_lats[:]

    outVar = fh_out.createVariable('lon', 'f4', ('lon',))
    outVar.setncatts({k: origi_lons.getncattr(k) for k in origi_lons.ncattrs()})
    outVar[:] = ref_lons[:]

    outVar = fh_out.createVariable('county_label', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    outVar[:] = ma.masked_equal(sampled_values['county_label'], 0)

    outVar = fh_out.createVariable('state_code', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    outVar[:] = ma.masked_equal(sampled_values['state_code'], 0)

    outVar = fh_out.createVariable('county_code', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    outVar[:] = ma.masked_equal(sampled_values['county_code'], 0)

    fh_in.close()
    fh_ref.close()
    fh_out.close()


if __name__ == '__main__':
    reproject_us_counties('../../processed_data/counties/us_counties.nc',
                          '../../processed_data/lst/monthly_1km/201505.nc',
                          '../../processed_data/counties/lst/us_counties.nc')
