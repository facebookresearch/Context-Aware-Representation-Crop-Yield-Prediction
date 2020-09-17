#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from data_preprocessing import CLIMATE_VARS
from data_preprocessing.sample_quadruplets import generate_training_for_counties

from netCDF4 import Dataset
import pandas as pd
import numpy.ma as ma
from collections import defaultdict
import numpy as np


def generate_dims_for_counties(croptype):
    yield_data = pd.read_csv('processed_data/crop_yield/{}_2000_2018.csv'.format(croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    ppt_fh = Dataset('experiment_data/spatial_temporal/nc_files/2014.nc', 'r')
    v_ppt = ppt_fh.variables['ppt'][0, :, :]
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})
    counties = pd.read_csv('processed_data/counties/lst/us_counties_cro_cvm_locations.csv')
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat, lon, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat, c.lon, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat, lon, lat0, lat1, lon0, lon1]

    yield_dim_csv = []
    for yd in yield_data.itertuples():
        year, state, county, value = yd.year, yd.state, yd.county, yd.value

        if (state, county) not in county_dic:
            continue

        lat, lon, lat0, lat1, lon0, lon1 = county_dic[(state, county)]
        assert lat1 - lat0 == 49
        assert lon1 - lon0 == 49

        selected_ppt = v_ppt[lat0:lat1+1, lon0:lon1+1]
        if ma.count_masked(selected_ppt) != 0:
            continue

        yield_dim_csv.append([state, county, year, value, lat, lon])

    yield_dim_csv = pd.DataFrame(yield_dim_csv, columns=['state', 'county', 'year', 'value', 'lat', 'lon'])
    yield_dim_csv.to_csv('experiment_data/spatial_temporal/counties/deep_gaussian_dim_y.csv')


def get_max_min_val_for_climate_variable(img_dir):
    cv_dic = defaultdict(list)

    for year in range(2000, 2014):
        fh = Dataset('{}/{}.nc'.format(img_dir, year))

        for v_name, varin in fh.variables.items():
            if v_name in CLIMATE_VARS:
                cv_dic[v_name].append(varin[:].compressed())
        fh.close()

    for cv in CLIMATE_VARS:
        values = np.asarray(cv_dic[cv])
        print(cv, np.percentile(values, 95))
        print(cv, np.percentile(values, 5))


if __name__ == '__main__':
    generate_dims_for_counties(croptype='soybeans')
    get_max_min_val_for_climate_variable('experiment_data/spatial_temporal/nc_files')
    for nr in [25]:
    # for nr in [10, 25, 50, 100, 500, None]:
        generate_training_for_counties(out_dir='experiment_data/deep_gaussian/counties',
                                       img_dir='experiment_data/spatial_temporal/nc_files',
                                       start_month=3, end_month=9, start_month_index=1, n_spatial_neighbor=1, n_distant=1,
                                       img_timestep_quadruplets=
                                       'experiment_data/spatial_temporal/counties/img_timestep_quadruplets_hard.csv',
                                       img_size=50, neighborhood_radius=nr, prenorm=False)
