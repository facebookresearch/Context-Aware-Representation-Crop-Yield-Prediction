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


def reproject_elevation():
    fh_in = Dataset('../../raw_data/elevation/90m.nc', 'r')
    fh_out = Dataset('../../processed_data/elevation/1km.nc', 'w')
    fh_ref = Dataset('../../processed_data/lst/monthly_1km/201508.nc', 'r')

    ref_lats, ref_lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]
    lat_bins, lon_bins = get_lat_lon_bins(ref_lats, ref_lons)

    ele_lats = fh_in.variables['lat']
    ele_lats_value = ele_lats[:][::-1]
    ele_lons = fh_in.variables['lon']
    ele_lons_value = ele_lons[:]
    ele_var = fh_in.variables['Band1'][0, :, :]
    ele_resampled = np.full((len(ref_lats), len(ref_lons)), -9999.9)
    # ele_std_resampled = np.full((len(ref_lats), len(ref_lons)), -9999.9)

    for id_lats in range(len(ref_lats)):
        for id_lons in range(len(ref_lons)):
            lats_index = np.searchsorted(ele_lats_value, [lat_bins[id_lats + 1], lat_bins[id_lats]])
            lons_index = np.searchsorted(ele_lons_value, [lon_bins[id_lons], lon_bins[id_lons + 1]])
            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                ele_selected = ele_var[np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                       np.array(range(lons_index[0], lons_index[1]))]
                avg = ma.mean(ele_selected)
                # std = ma.std(ele_selected)
                ele_resampled[id_lats, id_lons] = (avg if avg is not ma.masked else -9999.9)
                # ele_std_resampled[id_lats, id_lons] = (std if std is not ma.masked else -9999.9)
        print(id_lats)

    ele_resampled = ma.masked_equal(ele_resampled, -9999.9)
    # ele_std_resampled = ma.masked_equal(ele_std_resampled, -9999.9)

    fh_out.createDimension('lat', len(ref_lats))
    fh_out.createDimension('lon', len(ref_lons))

    outVar = fh_out.createVariable('lat', 'f4', ('lat',))
    outVar.setncatts({k: ele_lats.getncattr(k) for k in ele_lats.ncattrs()})
    outVar[:] = ref_lats[:]

    outVar = fh_out.createVariable('lon', 'f4', ('lon',))
    outVar.setncatts({k: ele_lons.getncattr(k) for k in ele_lons.ncattrs()})
    outVar[:] = ref_lons[:]

    # outVar = fh_out.createVariable('elevation_mean', 'f4', ('lat', 'lon',))
    outVar = fh_out.createVariable('elevation', 'f4', ('lat', 'lon',))
    outVar.setncatts({'units': "m"})
    outVar.setncatts({'long_name': "USGS_NED Elevation value"})
    outVar.setncatts({'_FillValue': np.array([-9999.9]).astype('f')})
    outVar[:] = ele_resampled[:]

    # outVar = fh_out.createVariable('elevation_std', 'f4', ('lat', 'lon',))
    # outVar.setncatts({'units': "m"})
    # outVar.setncatts({'long_name': "USGS_NED Elevation value"})
    # outVar.setncatts({'_FillValue': np.array([-9999.9]).astype('f')})
    # outVar[:] = ele_std_resampled[:]


if __name__ == '__main__':
    reproject_elevation()
