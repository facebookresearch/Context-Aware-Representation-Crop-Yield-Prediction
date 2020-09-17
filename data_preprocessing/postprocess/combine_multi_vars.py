#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
import csv
from collections import defaultdict
import numpy.ma as ma
import pickle
import sys

sys.path.append("..")

from data_preprocessing import CLIMATE_VARS
from data_preprocessing import STATIC_CLIMATE_VARS
from data_preprocessing import DYNAMIC_CLIMATE_VARS


def get_counties_lat_lon_indices(in_file, out_file, n_pixels):
    counties = pd.read_csv(in_file)
    counties.columns = ['state', 'county', 'lat', 'lon']
    ref_file = Dataset('../../processed_data/lst/monthly_1km/201505.nc', 'r')
    ref_lats, ref_lons = ref_file.variables['lat'][:], ref_file.variables['lon'][:]
    ref_file.close()

    # output columns: state, county, lat, lon, lat_index0, lat_index1, lon_index0, lon_index1
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['state', 'county', 'lat', 'lon', 'lat0', 'lat1', 'lon0', 'lon1'])
        for c in counties.itertuples():
            state, county, lat, lon = c.state, c.county, c.lat, c.lon
            lat_indices = sorted(np.argsort(np.abs(ref_lats - lat))[:n_pixels])
            lon_indices = sorted(np.argsort(np.abs(ref_lons - lon))[:n_pixels])

            line = [state, county, lat, lon, lat_indices[0], lat_indices[-1], lon_indices[0], lon_indices[-1]]
            writer.writerow(line)


def combine_by_year(start_month, end_month, dir_var_tuple_list):
    fill_value_dic = {'ndvi': -0.2, 'evi': -0.2, 'elevation': 0,
                      'lst_day': 280, 'lst_night': 280, 'sand': 101, 'clay': 101, 'silt': 101}

    for year in range(2000, 2018):
        fh_out = Dataset('../../experiment_data/spatial_temporal/nc_files_unmasked/{}.nc'.format(year), 'w')

        var_list = []
        n_t = end_month - start_month + 1
        first_first_flag = True
        first_flag = True
        n_dim = {}
        ppt_mask = None

        for i_month, month in enumerate(range(start_month, end_month+1)):
            for (f_dir, selected_vars) in dir_var_tuple_list:
                if os.path.isfile(f_dir):
                    fh_in = Dataset(f_dir, 'r')
                else:
                    fh_in = Dataset('{}/{}{}.nc'.format(f_dir, year, '{0:02}'.format(month)))

                if first_first_flag:
                    for name, dim in fh_in.dimensions.items():
                        n_dim[name] = len(dim)
                        fh_out.createDimension(name, len(dim))

                    fh_out.createDimension('time', n_t)
                    outVar = fh_out.createVariable('time', 'int', ("time",))
                    outVar[:] = range(start_month, end_month + 1)

                    for v_name, varin in fh_in.variables.items():
                        if v_name == 'lat' or v_name == 'lon':
                            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                            outVar[:] = varin[:]

                    first_first_flag = False

                if first_flag:
                    for v_name, varin in fh_in.variables.items():
                        if v_name in selected_vars:
                            var_list.append(v_name)
                            outVar = fh_out.createVariable(v_name, 'f4', ("time", "lat", "lon",))
                            # outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                            outVar[:] = ma.empty((n_t, n_dim['lat'], n_dim['lon']))
                        if v_name == 'ppt':
                            ppt_mask = ma.getmaskarray(fh_in.variables['ppt'][:])

                assert ppt_mask is not None
                for vname in selected_vars:
                    if vname != 'ppt':
                        var_value = ma.filled(fh_in.variables[vname][:], fill_value=fill_value_dic[vname])
                        var_value = ma.array(var_value, mask=ppt_mask)
                    else:
                        var_value = fh_in.variables[vname][:]
                    fh_out.variables[vname][i_month, :, :] = var_value
                fh_in.close()

            first_flag = False

        print(var_list)
        fh_out.close()


def generate_no_spatial_for_counties(yield_data_dir, ppt_file, county_location_file, out_dir, img_dir, croptype, start_month, end_month, start_index):
    yield_data = pd.read_csv('{}/{}_2000_2018.csv'.format(yield_data_dir, croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})

    ppt_fh = Dataset(ppt_file, 'r')
    v_ppt = ppt_fh.variables['ppt'][0, :, :]

    counties = pd.read_csv(county_location_file)
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat0, lat1, lon0, lon1]

    csv_header = ['year', 'state', 'county', 'yield']
    for climate_var in DYNAMIC_CLIMATE_VARS:
        for month in map(str, range(start_month, end_month+1)):
            csv_header.append(climate_var + "_" + month)
    for climate_var in STATIC_CLIMATE_VARS:
        csv_header.append(climate_var)
    for climate_var in CLIMATE_VARS:
        csv_header.append(climate_var + "_mean")

    output_file = '{}/{}_{}_{}.csv'.format(out_dir, croptype, start_month, end_month)
    n_t = end_month - start_month + 1
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        for yd in yield_data.itertuples():
            year, state, county, value = yd.year, yd.state, yd.county, yd.value

            # no location info
            if (state, county) not in county_dic:
                continue
            lat0, lat1, lon0, lon1 = county_dic[(state, county)]
            assert lat1 - lat0 == 49
            assert lon1 - lon0 == 49

            selected_ppt = v_ppt[lat0:lat1 + 1, lon0:lon1 + 1]
            if ma.count_masked(selected_ppt) != 0:
                continue

            values = [year, state, county, value]
            value_dic = defaultdict(list)

            if '{}.nc'.format(year) not in os.listdir(img_dir):
                continue

            fh = Dataset('{}/{}.nc'.format(img_dir, year))
            for climate_var in DYNAMIC_CLIMATE_VARS:
                for i_month in range(n_t):
                    selected_values = fh.variables[climate_var][i_month+start_index, lat0:lat1+1, lon0:lon1+1]
                    averaged = ma.mean(selected_values)
                    values.append(averaged)
                    value_dic[climate_var].append(averaged)
            for climate_var in STATIC_CLIMATE_VARS:
                selected_values = fh.variables[climate_var][0, lat0:lat1 + 1, lon0:lon1 + 1]
                averaged = ma.mean(selected_values)
                values.append(averaged)
                value_dic[climate_var].append(averaged)
            fh.close()

            for climate_var in CLIMATE_VARS:
                values.append(np.mean(value_dic[climate_var]))

            writer.writerow(values)


def mask_non_major_states(in_dir, out_dir, mask_file, major_states):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fh_mask = Dataset(mask_file, 'r')
    state_codes = fh_mask.variables['state_code'][:]
    major_state_mask = ~np.in1d(state_codes, major_states).reshape(state_codes.shape)

    for nc_file in os.listdir(in_dir):
        if nc_file.endswith('.nc'):
            fh_in = Dataset('{}/{}'.format(in_dir, nc_file), 'r')
            fh_out = Dataset('{}/{}'.format(out_dir, nc_file), 'w')

            for name, dim in fh_in.dimensions.items():
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

            for v_name, varin in fh_in.variables.items():
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                if v_name in ['lat', 'lon', 'time']:
                    outVar[:] = varin[:]
                else:
                    if 'time' not in fh_in.variables:
                        outVar[:] = ma.array(varin[:], mask=major_state_mask)
                    else:
                        outVar[:] = ma.array(varin[:], mask=np.tile(major_state_mask, (varin[:].shape[0], 1)))
            fh_in.close()
            fh_out.close()

    fh_mask.close()


def obtain_channel_wise_mean_std(img_dir):
    mean_dic = {}
    std_dic = {}
    for month_index in range(8):
        cv_dic = defaultdict(list)

        for year in range(2000, 2014):
            fh = Dataset('{}/{}.nc'.format(img_dir, year))

            for v_name, varin in fh.variables.items():
                if v_name in CLIMATE_VARS:
                    cv_dic[v_name].append(varin[month_index].compressed())
            fh.close()

        means = []
        stds = []
        for cv in CLIMATE_VARS:
            values = np.asarray(cv_dic[cv])
            means.append(np.mean(values))
            stds.append(np.std(values))
        mean_dic[month_index] = means
        std_dic[month_index] = stds

    with open('{}/monthly_channel_wise_mean.pkl'.format(img_dir), 'wb') as f:
        pickle.dump(mean_dic, f)
    with open('{}/monthly_channel_wise_std.pkl'.format(img_dir), 'wb') as f:
        pickle.dump(std_dic, f)


if __name__ == '__main__':
    # get_counties_lat_lon_indices('../../processed_data/counties/us_counties_cro_cvm_locations.csv',
    #                              '../../processed_data/counties/lst/us_counties_cro_cvm_locations.csv',
    #                              n_pixels=50)

    # combine_by_year(start_month=2,
    #                 end_month=9,
    #                 dir_var_tuple_list=[
    #                     ('../../processed_data/prism/combined_monthly_1km', ['ppt']),
    #                     ('../../processed_data/ndvi/1km', ['ndvi', 'evi']),
    #                     ('../../processed_data/elevation/1km.nc', ['elevation']),
    #                     ('../../processed_data/lst/monthly_1km', ['lst_day', 'lst_night']),
    #                     ('../../processed_data/soil_fraction/soil_fraction_usa_1km.nc', ['sand', 'clay', 'silt'])])

    generate_no_spatial_for_counties(yield_data_dir='../../processed_data/crop_yield',
                                     county_location_file='../../processed_data/counties/lst/us_counties_cro_cvm_locations.csv',
                                     out_dir='../../experiment_data/no_spatial',
                                     img_dir='../../experiment_data/spatial_temporal/nc_files',
                                     croptype='soybeans',
                                     start_month=3,
                                     end_month=9,
                                     start_index=1)
    MAJOR_STATES = [1,  5, 10, 13, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 31, 34, 36, 37, 38, 39, 40, 42, 45,
                    46, 47, 48, 51, 55, 54, 12]
    mask_non_major_states('../../experiment_data/spatial_temporal/nc_files_unmasked',
                          '../../experiment_data/spatial_temporal/nc_files',
                          '../../processed_data/counties/lst/us_counties.nc',
                          MAJOR_STATES)
