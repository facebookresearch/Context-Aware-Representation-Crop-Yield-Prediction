#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
import csv
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from collections import defaultdict
import os


def get_counties_lat_lon_indices(in_file, out_file, n_pixels):
    counties = pd.read_csv(in_file)
    counties.columns = ['state', 'county', 'lat', 'lon']
    prism_file = Dataset('../../processed_data/prism/monthly/ppt_199901.nc', 'r')
    prism_lats, prism_lons = prism_file.variables['lat'][:], prism_file.variables['lon'][:]
    prism_file.close()

    # output columns: state, county, lat, lon, lat_index0, lat_index1, lon_index0, lon_index1
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['state', 'county', 'lat', 'lon', 'lat0', 'lat1', 'lon0', 'lon1'])
        for c in counties.itertuples():
            state, county, lat, lon = c.state, c.county, c.lat, c.lon
            lat_indices = sorted(np.argsort(np.abs(prism_lats - lat))[:n_pixels])
            lon_indices = sorted(np.argsort(np.abs(prism_lons - lon))[:n_pixels])

            line = [state, county, lat, lon, lat_indices[0], lat_indices[-1], lon_indices[0], lon_indices[-1]]
            writer.writerow(line)


def generate_no_spatial(croptype, start_month, end_month, selected_states=None):
    yield_data = pd.read_csv('../../processed_data/crop_yield/{}_1999_2018.csv'.format(croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})
    counties = pd.read_csv('../../processed_data/counties/prism/us_counties_cro_cvm_locations.csv')
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat0, lat1, lon0, lon1]

    climate_vars = ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]
    csv_header = ['year', 'state', 'county', 'yield']
    for month in map(str, range(start_month, end_month+1)):
        for climate_var in climate_vars:
            csv_header.append(climate_var + "_" + month)
    for climate_var in climate_vars:
        csv_header.append(climate_var + "_mean")

    output_file = '../../experiment_data/no_spatial/{}_{}_{}.csv'.format(croptype, start_month, end_month) \
        if not selected_states else '../../experiment_data/no_spatial/{}_{}_{}_major_states.csv'.format(croptype, start_month, end_month)
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        for yd in yield_data.itertuples():
            year, state, county, value = yd.year, yd.state, yd.county, yd.value

            if selected_states is not None and state not in selected_states:
                continue
            # no location info
            if (state, county) not in county_dic:
                continue
            lat0, lat1, lon0, lon1 = county_dic[(state, county)]
            assert lat1 - lat0 == 9
            assert lon1 - lon0 == 9

            values = [year, state, county, value]
            value_dic = defaultdict(list)
            for month in range(start_month, end_month+1):
                fh = Dataset('../../processed_data/prism/combined_monthly/{}{}.nc'.format(year, '{0:02}'.format(month)))

                for climate_var in climate_vars:
                    selected_values = fh.variables[climate_var][lat0:lat1+1, lon0:lon1+1]
                    averaged = ma.mean(selected_values)
                    values.append(averaged)
                    value_dic[climate_var].append(averaged)
                fh.close()

            for climate_var in climate_vars:
                values.append(np.mean(value_dic[climate_var]))

            writer.writerow(values)


def average_by_year(start_month, end_month):
    if not os.path.exists('../../experiment_data/only_spatial/averaged_{}_{}/nc'.format(start_month, end_month)):
        os.makedirs('../../experiment_data/only_spatial/averaged_{}_{}/nc'.format(start_month, end_month))
    for year in range(1999, 2019):
        fh_out = Dataset('../../experiment_data/only_spatial/averaged_{}_{}/nc/{}.nc'.format(start_month, end_month, year), 'w')

        first_flag = True
        var_lis = defaultdict(list)
        for month in range(start_month, end_month+1):
            fh_in = Dataset('../../processed_data/prism/combined_monthly/{}{}.nc'.format(year, '{0:02}'.format(month)))

            if first_flag:
                for name, dim in fh_in.dimensions.items():
                    fh_out.createDimension(name, len(dim))
                for v_name, varin in fh_in.variables.items():
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    if v_name in ["lat", "lon"]:
                        outVar[:] = varin[:]
                first_flag = False

            for v_name, varin in fh_in.variables.items():
                if v_name not in ["lat", "lon"]:
                    var_lis[v_name].append(fh_in.variables[v_name][:])

            fh_in.close()

        for var in fh_out.variables:
            if var != "lat" and var != "lon":
                fh_out.variables[var][:] = ma.array(var_lis[var]).mean(axis=0)

        fh_out.close()


def generate_only_spatial(croptype, start_month, end_month, selected_states=None):
    yield_data = pd.read_csv('../../processed_data/crop_yield/{}_1999_2018.csv'.format(croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})
    counties = pd.read_csv('../../processed_data/counties/prism/us_counties_cro_cvm_locations.csv')
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat0, lat1, lon0, lon1]

    climate_vars = ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]
    yield_values = []
    output_folder = 'counties' if not selected_states else 'counties_major_states'
    if not os.path.exists('../../experiment_data/only_spatial/averaged_{}_{}/{}'.format(start_month, end_month, output_folder)):
        os.makedirs('../../experiment_data/only_spatial/averaged_{}_{}/{}'.format(start_month, end_month, output_folder))
    for yd in yield_data.itertuples():
        year, state, county, value = yd.year, yd.state, yd.county, yd.value

        if selected_states is not None and state not in selected_states:
            continue
        # no location info
        if (state, county) not in county_dic:
            continue

        lat0, lat1, lon0, lon1 = county_dic[(state, county)]
        assert lat1 - lat0 == 9
        assert lon1 - lon0 == 9

        values = []
        fh = Dataset('../../experiment_data/only_spatial/averaged_{}_{}/nc/{}.nc'.format(start_month, end_month, year))
        for climate_var in climate_vars:
            values.append(fh.variables[climate_var][lat0:lat1+1, lon0:lon1+1])
        values = np.asarray(values)
        np.save('../../experiment_data/only_spatial/averaged_{}_{}/{}/{}_{}_{}.npy'.format(start_month, end_month, output_folder,
                                                                                     state, county, year), values)
        yield_values.append([year, state, county, value])
        assert values.shape == (7, 10, 10), values.shape
        fh.close()
    np.save('../../experiment_data/only_spatial/averaged_{}_{}/{}/y.npy'.format(start_month, end_month, output_folder),
            np.asarray(yield_values))


def combine_by_year(start_month, end_month):
    for year in range(1999, 2019):
        fh_out = Dataset('../../experiment_data/spatial_temporal/{}_{}/nc/{}.nc'.format(start_month, end_month, year), 'w')

        var_list = []
        n_t = end_month - start_month + 1
        first_flag = True
        n_dim = {}

        for i_month, month in enumerate(range(start_month, end_month+1)):
            fh_in = Dataset('../../processed_data/prism/combined_monthly/{}{}.nc'.format(year, '{0:02}'.format(month)))

            if first_flag:
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
                    else:
                        var_list.append(v_name)
                        outVar = fh_out.createVariable(v_name, varin.datatype, ("time", "lat", "lon",))
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = np.empty((n_t, n_dim['lat'], n_dim['lon']))

                first_flag = False

            for vname in var_list:
                var_value = fh_in.variables[vname][:]
                fh_out.variables[vname][i_month, :, :] = var_value[:]

            fh_in.close()

        fh_out.close()


def generate_spatial_temporal(croptype, start_month, end_month, selected_states=None):
    yield_data = pd.read_csv('../../processed_data/crop_yield/{}_1999_2018.csv'.format(croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})
    counties = pd.read_csv('../../processed_data/counties/prism/us_counties_cro_cvm_locations.csv')
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat0, lat1, lon0, lon1]

    climate_vars = ["ppt", "tdmean", "tmax", "tmean", "tmin", "vpdmax", "vpdmin"]
    yield_values = []
    output_folder = 'counties' if not selected_states else 'counties_major_states'
    if not os.path.exists(
            '../../experiment_data/spatial_temporal/{}_{}/{}'.format(start_month, end_month, output_folder)):
        os.makedirs(
            '../../experiment_data/spatial_temporal/{}_{}/{}'.format(start_month, end_month, output_folder))
    for yd in yield_data.itertuples():
        year, state, county, value = yd.year, yd.state, yd.county, yd.value

        if selected_states is not None and state not in selected_states:
            continue
        # no location info
        if (state, county) not in county_dic:
            continue

        lat0, lat1, lon0, lon1 = county_dic[(state, county)]
        assert lat1 - lat0 == 9
        assert lon1 - lon0 == 9

        values = []
        fh = Dataset('../../experiment_data/spatial_temporal/{}_{}/nc/{}.nc'.format(start_month, end_month, year))
        for climate_var in climate_vars:
            values.append(fh.variables[climate_var][:, lat0:lat1 + 1, lon0:lon1 + 1])
        values = np.asarray(values)
        np.save('../../experiment_data/spatial_temporal/{}_{}/{}/{}_{}_{}.npy'.format(start_month, end_month,
                                                                                      output_folder, state, county, year),
                values)
        yield_values.append([year, state, county, value])
        assert values.shape == (7, end_month-start_month+1, 10, 10), values.shape
        fh.close()
    np.save('../../experiment_data/spatial_temporal/{}_{}/{}/y.npy'.format(start_month, end_month, output_folder),
            np.asarray(yield_values))


if __name__ == '__main__':
    # get_counties_lat_lon_indices('../../processed_data/counties/us_counties_cro_cvm_locations.csv',
    #                              '../../processed_data/counties/prism/us_counties_cro_cvm_locations.csv',
    #                              n_pixels=10)
    #
    # average_by_year(1, 9)
    # combine_by_year(1, 9)

    # generate_no_spatial('soybeans', 1, 9)
    generate_only_spatial('soybeans', 1, 9)
    generate_spatial_temporal('soybeans', 1, 9)

    MAJOR_STATES = [5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46]
    # mask_non_major_states('../../experiment_data/only_spatial/averaged_1_9/nc',
    #                       '../../experiment_data/only_spatial/averaged_1_9/nc_major_states',
    #                       MAJOR_STATES)
    # mask_non_major_states('../../experiment_data/spatial_temporal/1_9/nc',
    #                       '../../experiment_data/spatial_temporal/1_9/nc_major_states',
    #                       MAJOR_STATES)
    # generate_no_spatial('soybeans', 1, 9, selected_states=MAJOR_STATES)
    generate_only_spatial('soybeans', 1, 9, selected_states=MAJOR_STATES)
    generate_spatial_temporal('soybeans', 1, 9, selected_states=MAJOR_STATES)
