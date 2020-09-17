#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from netCDF4 import Dataset
import datetime
import calendar
from collections import defaultdict
import numpy.ma as ma
import os

import sys
sys.path.append("..")


def extract_lst(nc_file):
    fh_in = Dataset('../../raw_data/lst/' + nc_file, 'r')

    for index, n_days in enumerate(fh_in.variables['time'][:]):
        date = (datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(int(n_days))).strftime('%Y%m%d')
        print(date)
        fh_out = Dataset('../../raw_data/lst/1km/{}.nc'.format(date), 'w')

        for name, dim in fh_in.dimensions.items():
            if name != 'time':
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

        ignore_features = ['time', 'crs', 'Clear_day_cov', 'Clear_night_cov', 'Day_view_angl', 'Day_view_time',
                           'Night_view_angl', 'Night_view_time', 'Emis_31', 'Emis_32', "QC_Day", "QC_Night"]
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
                    outVar[:] = varin[index, :, :]
        fh_out.close()
    fh_in.close()


def generate_monthly_average(start_year, end_year, start_month, end_month):
    in_dir = '../../raw_data/lst/1km'
    out_dir = '../../processed_data/lst/monthly_1km'
    os.makedirs(out_dir, exist_ok=True)
    for year in range(start_year, end_year):
        for month in range(start_month, end_month):
            fh_out = Dataset('{}/{}{}.nc'.format(out_dir, year, '{0:02}'.format(month)), 'w')
            print(year, month)

            var_lis = defaultdict(list)
            first = True
            num_days = calendar.monthrange(year, month)[1]
            days = map(lambda x: x.strftime('%Y%m%d'), [datetime.date(year, month, day) for day in range(1, num_days+1)])
            for day in days:
                if '{}.nc'.format(day) not in os.listdir(in_dir):
                    print('Missing {}'.format(day))
                    continue
                fh_in = Dataset('{}/{}.nc'.format(in_dir, day), 'r')

                len_lat, len_lon = len(fh_in.variables['lat'][:]), len(fh_in.variables['lon'][:])
                assert len_lat == 3578 or len_lat == 3579
                assert len_lon == 7797

                for v_name, varin in fh_in.variables.items():
                    if v_name in ['LST_Day_1km', 'LST_Night_1km']:
                        if len_lat == 3578:
                            var_lis[v_name[:-4].lower()].append(fh_in.variables[v_name][:])
                        else:
                            var_lis[v_name[:-4].lower()].append(fh_in.variables[v_name][:-1, :])

                if first:
                    for name, dim in fh_in.dimensions.items():
                        if name == 'lat':
                            fh_out.createDimension(name, 3578)
                        else:
                            fh_out.createDimension(name, len(dim))
                    for v_name, varin in fh_in.variables.items():
                        if v_name in ['LST_Day_1km', 'LST_Night_1km'] or v_name in ["lat", "lon"]:
                            new_name = v_name[:-4].lower() if v_name in ['LST_Day_1km', 'LST_Night_1km'] else v_name
                            outVar = fh_out.createVariable(new_name, varin.datatype, varin.dimensions)
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                            if v_name == 'lat':
                                outVar[:] = varin[:3578]
                            elif v_name == 'lon':
                                outVar[:] = varin[:]

                    first = False

                fh_in.close()

            for var in fh_out.variables:
                if var != "lat" and var != "lon":
                    print(ma.array(var_lis[var]).shape)
                    fh_out.variables[var][:] = ma.array(var_lis[var]).mean(axis=0)

            fh_out.close()


if __name__ == '__main__':
    extract_lst('MOD11A1_20140201_20140930.nc')
    generate_monthly_average(2014, 2015, 2, 10)
