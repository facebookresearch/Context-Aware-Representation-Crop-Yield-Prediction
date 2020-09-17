#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import numpy.ma as ma
import datetime

import sys
sys.path.append("..")


def extract_lai(nc_file):
    fh_in = Dataset('../../raw_data/lai/' + nc_file, 'r')

    for index, n_days in enumerate(fh_in.variables['time'][:]):
        date = (datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(int(n_days))).strftime('%Y%m%d')
        print(date)
        fh_out = Dataset('../../processed_data/lai/500m/{}.nc'.format(date), 'w')

        for name, dim in fh_in.dimensions.items():
            if name != 'time':
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

        ignore_features = ["time", "crs", "FparExtra_QC", "FparLai_QC"]
        mask_value_dic = {'Lai_500m': 10, 'LaiStdDev_500m': 10, 'Fpar_500m': 1, 'FparStdDev_500m': 1}
        for v_name, varin in fh_in.variables.items():
            if v_name not in ignore_features:
                dimensions = varin.dimensions if v_name in ['lat', 'lon'] else ('lat', 'lon')
                outVar = fh_out.createVariable(v_name, varin.datatype, dimensions)
                if v_name == "lat":
                    outVar.setncatts({"units": "degree_north"})
                    outVar[:] = varin[:]
                elif v_name == "lon":
                    outVar.setncatts({"units": "degree_east"})
                    outVar[:] = varin[:]
                else:
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    vin = varin[index, :, :]
                    vin = ma.masked_greater(vin, mask_value_dic[v_name])
                    vin = ma.masked_less(vin, 0)
                    outVar[:] = vin[:]
        fh_out.close()
    fh_in.close()


def extract_ndvi(nc_file):
    fh_in = Dataset('../../raw_data/ndvi/' + nc_file, 'r')

    for index, n_days in enumerate(fh_in.variables['time'][:]):
        date = (datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(int(n_days))).strftime('%Y%m%d')
        print(date)
        fh_out = Dataset('../../processed_data/ndvi/1km/{}.nc'.format(date[:-2]), 'w')

        for name, dim in fh_in.dimensions.items():
            if name != 'time':
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

        ignore_features = ["time", "crs", "_1_km_monthly_VI_Quality"]
        for v_name, varin in fh_in.variables.items():
            if v_name not in ignore_features:
                dimensions = varin.dimensions if v_name in ['lat', 'lon'] else ('lat', 'lon')
                v_name = v_name if v_name in ['lat', 'lon'] else v_name.split('_')[-1].lower()
                outVar = fh_out.createVariable(v_name, varin.datatype, dimensions)
                if v_name == "lat":
                    outVar.setncatts({"units": "degree_north"})
                    outVar[:] = varin[:]
                elif v_name == "lon":
                    outVar.setncatts({"units": "degree_east"})
                    outVar[:] = varin[:]
                else:
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    vin = varin[index, :, :]
                    vin = ma.masked_greater(vin, 1.0)
                    vin = ma.masked_less(vin, -0.2)
                    outVar[:] = vin[:]
        fh_out.close()
    fh_in.close()


if __name__ == '__main__':
    # extract_lai('20190604.nc')
    extract_ndvi('MOD13A3_20000201_20181231.nc')
