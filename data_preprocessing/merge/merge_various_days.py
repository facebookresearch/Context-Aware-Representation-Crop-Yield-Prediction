#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..utils import generate_doy

import os
import numpy as np
import datetime as dt
from datetime import datetime
from netCDF4 import Dataset

FIRST_DATE = dt.date(2001, 1, 1)


def merge_various_days(in_path, out_path, fout_name, doy_start=None, doy_end=None, select_vars=None):
    fh_out = Dataset(os.path.join(out_path, fout_name + '.nc'), 'w')

    num = 0
    var_list = []

    if doy_start is None or doy_end is None:
        fnames = [fname[:-3] for fname in os.listdir(in_path) if fname.endswith(".nc")]
        fnames = sorted(fnames, key=lambda x: datetime.strptime("".join(c for c in x if c.isdigit()), '%Y%m%d'))
    else:
        fnames = list(generate_doy(doy_start, doy_end, ""))
    num_files = len(fnames)
    print("Number of files", num_files)

    for nc_file in fnames:
        nc_doy = "".join(c for c in nc_file if c.isdigit())
        fh_in = Dataset(os.path.join(in_path, nc_file + ".nc"), 'r')
        n_dim = {}
        if num == 0:
            for name, dim in fh_in.dimensions.items():
                n_dim[name] = len(dim)
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

            fh_out.createDimension('time', num_files)
            outVar = fh_out.createVariable('time', 'int', ("time",))
            outVar[:] = range(1, num_files + 1)

            select_vars = list(fh_in.variables.keys()) if select_vars is None else select_vars
            for v_name, varin in fh_in.variables.items():
                if v_name == 'lat' or v_name == 'lon':
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
                else:
                    if v_name in select_vars:
                        var_list.append(v_name)
                        outVar = fh_out.createVariable(v_name, varin.datatype, ("time", "lat", "lon",))
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = np.empty((num_files, n_dim['lat'], n_dim['lon']))

        current_date = datetime.strptime(nc_doy, "%Y%m%d").date()
        fh_out.variables['time'][num] = (current_date - FIRST_DATE).days
        for vname in var_list:
            var_value = fh_in.variables[vname][:]
            fh_out.variables[vname][num, :, :] = var_value[:]

        num += 1
        fh_in.close()
    fh_out.close()

    print(num, num_files)
    assert (num == num_files)
