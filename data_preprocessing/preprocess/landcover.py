#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import sys
sys.path.append("..")

from data_preprocessing.merge import merge_various_days


def generate_convert_to_nc_script():
    fh_out = open('../../processed_data/landcover/convert_to_nc.sh', 'w')
    fh_out.write('#!/bin/bash\n')

    for tif_file in os.listdir('../../processed_data/landcover/'):
        if tif_file.endswith('.tif'):
            fh_out.write('gdal_translate -of netCDF {} {}.nc\n'.format(tif_file, tif_file[:-4]))


def mask_with_landcover(out_folder, kept_ldcs):
    for nc_file in os.listdir('../../processed_data/landcover/origi/'):
        if nc_file.endswith('.nc'):
            fh_in = Dataset('../../processed_data/landcover/origi/{}'.format(nc_file), 'r')
            fh_out = Dataset('../../processed_data/landcover/{}/{}'.format(out_folder, nc_file), 'w')

            for name, dim in fh_in.dimensions.items():
                fh_out.createDimension(name, len(dim))

            for v_name, varin in fh_in.variables.items():
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                if v_name in ['lat', 'lon']:
                    outVar[:] = varin[:]
                else:
                    landcovers = varin[:]
                    lc_mask = np.in1d(landcovers, kept_ldcs).reshape(landcovers.shape)
                    outVar[:] = ma.array(varin[:], mask=~lc_mask)

            fh_in.close()
            fh_out.close()


def generate_cropland(in_file, out_file):
    fh_in = Dataset(in_file, 'r')
    fh_out = Dataset(out_file, 'w')

    lats, lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]
    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        if v_name in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    outVar = fh_out.createVariable('cropland', 'f4', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0.0]).astype('f')})
    cropland = np.full((len(lats), len(lons)), 1.0)
    mask_value = ma.getmaskarray(fh_in.variables['Band1'][:])
    mask_value = np.logical_and.reduce(mask_value)
    outVar[:] = ma.array(cropland, mask=mask_value)

    fh_in.close()
    fh_out.close()


if __name__ == '__main__':
    generate_convert_to_nc_script()
    mask_with_landcover('cro', [12])
    mask_with_landcover('cro_cvm', [12, 14])
    merge_various_days('../../processed_data/landcover/origi/', '../../processed_data/landcover/', 'ts_merged',
                       select_vars=['Band1'])
    merge_various_days('../../processed_data/landcover/cro/', '../../processed_data/landcover/', 'ts_merged_cro',
                       select_vars=['Band1'])
    merge_various_days('../../processed_data/landcover/cro_cvm/', '../../processed_data/landcover/',
                       'ts_merged_cro_cvm', select_vars=['Band1'])
    generate_cropland('../../processed_data/landcover/ts_merged_cro.nc',
                      '../../processed_data/landcover/cropland_cro.nc')
    generate_cropland('../../processed_data/landcover/ts_merged_cro_cvm.nc',
                      '../../processed_data/landcover/cropland_cro_cvm.nc')
