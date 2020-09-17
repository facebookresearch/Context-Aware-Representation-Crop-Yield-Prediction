#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from netCDF4 import Dataset

# gdal_translate -of netCDF PRISM_ppt_stable_4kmM3_201806_bil.bil PRISM_ppt_stable_4kmM3_201806.nc


def prism_convert_to_nc():
    fh_out = open(os.path.join("../..", "prism_convert_to_nc.sh"), "w")
    fh_out.write("#!/bin/bash\n")
    # m_dic = {"ppt": "M3", "tdmean": "M1", "tmax": "M2", "tmean": "M2", "tmin": "M2", "vpdmax": "M1", "vpdmin": "M1"}
    for climate_var in ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]:
        for year in range(1999, 2019):
            for month in range(1, 13):
                fh_out.write("gdal_translate -of netCDF raw_data/prism/monthly/PRISM_{}_stable_4kmM3_198101_201904_bil/"
                             "PRISM_{}_stable_4kmM3_{}{}_bil.bil processed_data/prism/monthly/{}_{}{}.nc\n"
                             .format(climate_var, climate_var, year, "{0:02}".format(month), climate_var,  year,
                                     "{0:02}".format(month)))


def combine_multivar():
    climate_vars = ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]

    for year in range(1999, 2019):
        for month in range(1, 13):
            fh_out = Dataset('../../processed_data/prism/combined_monthly/{}{}.nc'.format(year,
                                                                                          '{0:02}'.format(month)), 'w')
            first_flag = True
            for v in climate_vars:
                fh_in = Dataset('../../processed_data/prism/monthly/{}_{}{}.nc'.format(v, year,
                                                                                       '{0:02}'.format(month), 'r'))
                if first_flag:
                    for name, dim in fh_in.dimensions.items():
                        fh_out.createDimension(name, len(dim))
                    for v_name, varin in fh_in.variables.items():
                        if v_name in ['lat', 'lon']:
                            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                            outVar[:] = varin[:]
                    first_flag = False

                for v_name, varin in fh_in.variables.items():
                    if v_name == 'Band1':
                        outVar = fh_out.createVariable(v, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = varin[:]

                fh_in.close()

            fh_out.close()


if __name__ == "__main__":
    prism_convert_to_nc()
