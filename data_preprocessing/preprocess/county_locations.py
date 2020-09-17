#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append("..")

from data_preprocessing.utils import match_lat_lon
from data_preprocessing.plot import counties_plot


def generate_convert_to_nc_script():
    fh_out = open('../../processed_data/counties/all/convert_to_nc.sh', 'w')
    fh_out.write('#!/bin/bash\n')

    for tif_file in os.listdir('../../processed_data/counties/all/tif/'):
        if tif_file.endswith('.tif'):
            fh_out.write('gdal_translate -of netCDF tif/{} nc/{}.nc\n'.format(tif_file, tif_file[:-4]))


def combine_ncs():
    fh_out = Dataset('../../processed_data/counties/us_counties.nc', 'w')
    fh_ref = Dataset('../../processed_data/landcover/cropland_cro.nc', 'r')

    lats, lons = fh_ref.variables['lat'][:], fh_ref.variables['lon'][:]

    for name, dim in fh_ref.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_ref.variables.items():
        if v_name in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    outVar = fh_out.createVariable('county_label', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    counties_labels = np.full((len(lats), len(lons)), 0)

    outVar = fh_out.createVariable('state_code', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    state_code = np.full((len(lats), len(lons)), 0)

    outVar = fh_out.createVariable('county_code', 'int', ('lat', 'lon'))
    outVar.setncatts({'_FillValue': np.array([0]).astype(int)})
    county_code = np.full((len(lats), len(lons)), 0)

    for nc_file in os.listdir('../../processed_data/counties/all/nc/'):
        if nc_file.endswith('.nc'):
            # ignore Alaska
            if nc_file.split('_')[0] == '2':
                continue
            print(nc_file)
            fh_in = Dataset('../../processed_data/counties/all/nc/{}'.format(nc_file), 'r')
            local_lats, local_lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]

            i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(lats, lons, local_lats, local_lons)

            local_values = ma.masked_equal(fh_in.variables['Band1'][:], 0.0)
            for i, j in zip(*local_values.nonzero()):
                state, county = nc_file[:-3].split('_')
                state = str(state).zfill(2)
                county = str(county).zfill(3)
                counties_labels[i+i_lat_start, j+i_lon_start] = int(state+county)
                state_code[i+i_lat_start, j+i_lon_start] = int(state)
                county_code[i+i_lat_start, j+i_lon_start] = int(county)

            fh_in.close()

    fh_out.variables['county_label'][:] = ma.masked_equal(counties_labels, 0)
    fh_out.variables['state_code'][:] = ma.masked_equal(state_code, 0)
    fh_out.variables['county_code'][:] = ma.masked_equal(county_code, 0)

    fh_ref.close()
    fh_out.close()


def mask_with_landcover(out_file, ref_file):
    fh_in = Dataset('../../processed_data/counties/us_counties.nc', 'r')
    fh_out = Dataset(out_file, 'w')
    fh_ref = Dataset(ref_file, 'r')

    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        if v_name in ['lat', 'lon']:
            outVar[:] = varin[:]
        else:
            cropland_mask = ma.getmaskarray(fh_ref.variables['cropland'][:])
            outVar[:] = ma.array(varin[:], mask=cropland_mask)

    fh_in.close()
    fh_out.close()
    fh_ref.close()


def plot_counties(in_file):
    fh = Dataset(in_file, 'r')
    county_labels = fh.variables['county_label'][:]
    print(len(np.unique(county_labels.compressed())))
    fh.close()

    county_labels = np.unique(county_labels.compressed())
    county_labels = [[str(x).zfill(5)[:2], str(x).zfill(5)[2:]] for x in county_labels]

    data_dic = {}
    for state, county in county_labels:
        data_dic[state+county] = 100
    fake_quantiles = {x: 1 for x in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]}
    counties_plot(data_dic, Path('../../processed_data/counties/{}.html'.format(in_file[:-3])), fake_quantiles)

    return data_dic.keys()


def plot_counties_data(in_file):
    county_data = pd.read_csv(in_file)[['StateFips', 'CntyFips']]
    county_data.columns = ['State', 'County']

    data_dic = {}
    for row in county_data.itertuples():
        state, county = int(row.State), int(row.County)
        state = str(state).zfill(2)
        county = str(county).zfill(3)
        data_dic[state + county] = 100

    fake_quantiles = {x: 1 for x in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]}
    counties_plot(data_dic, Path('../../processed_data/counties/county_data.html'), fake_quantiles)

    return data_dic.keys()


def analyze_counties(in_file):
    fh = Dataset(in_file, 'r')
    counties, sizes = np.unique(fh.variables['county_label'][:].compressed(), return_counts=True)
    for county, size in zip(counties, sizes):
        print(county, size)
    plt.hist(sizes)
    plt.show()


def get_county_locations(in_file):
    fh = Dataset(in_file, 'r')
    lats, lons = fh.variables['lat'][:], fh.variables['lon'][:]

    county_labels = fh.variables['county_label'][:]
    counties = np.unique(county_labels.compressed())

    with open('{}_locations.csv'.format(in_file[:-3]), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['state', 'county', 'lat', 'lon'])
        for county in counties:
            selected_rows, selected_cols = np.where(county_labels == county)
            lat_mean, lon_mean = np.mean(lats[selected_rows]), np.mean(lons[selected_cols])
            line = [str(county).zfill(5)[:2], str(county).zfill(5)[2:], lat_mean, lon_mean]
            writer.writerow(line)


if __name__ == '__main__':
    # generate_convert_to_nc_script()
    combine_ncs()
    # mask_with_landcover('../../processed_data/counties/us_counties_cro.nc',
    #                     '../../processed_data/landcover/cropland_cro.nc')
    # mask_with_landcover('../../processed_data/counties/us_counties_cro_cvm.nc',
    #                     '../../processed_data/landcover/cropland_cro_cvm.nc')
    #
    # county_key = plot_counties_data('../../processed_data/counties/county_data.csv')
    # us_county_key = plot_counties('../../processed_data/counties/us_counties.nc')
    # print([x for x in us_county_key if x not in county_key])
    # print([x for x in county_key if x not in us_county_key and not x.startswith('02')])
    # plot_counties('../../processed_data/counties/us_counties_cro.nc')
    # plot_counties('../../processed_data/counties/us_counties_cro_cvm.nc')
    #
    # analyze_counties('../../processed_data/counties/us_counties.nc')
    # analyze_counties('../../processed_data/counties/us_counties_cro.nc')
    # analyze_counties('../../processed_data/counties/us_counties_cro_cvm.nc')

    # get_county_locations('../../processed_data/counties/us_counties.nc')
    # get_county_locations('../../processed_data/counties/us_counties_cro.nc')
    # get_county_locations('../../processed_data/counties/us_counties_cro_cvm.nc')
