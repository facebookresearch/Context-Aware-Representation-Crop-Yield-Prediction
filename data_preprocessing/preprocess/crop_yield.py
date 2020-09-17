#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from collections import defaultdict
import numpy as np
import numpy.ma as ma
from pathlib import Path
from netCDF4 import Dataset
from operator import itemgetter
import matplotlib.pyplot as plt
import sys
sys.path.append("..")


from data_preprocessing.plot import counties_plot, save_colorbar

state_dic = {10: 'Delaware', 1: 'Alabama', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii',
             16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 2: 'Alaska', 21: 'Kentucky',
             22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
             28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire',
             34: 'New Jersey', 35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio',
             40: 'Oklahoma', 4: 'Arizona', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina',
             46: 'South Dakota', 47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 5: 'Arkansas', 51: 'Virginia',
             53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming', 6: 'California', 8: 'Colorado',
             9: 'Connecticut'}


def clean_data(in_file, out_file, dropyear):
    yield_data = pd.read_csv(in_file)
    important_columns = ['Year', 'State ANSI', 'County ANSI', 'Value']
    yield_data = yield_data.dropna(subset=important_columns, how='any')
    yield_data = yield_data[yield_data.Year != 1999]
    yield_data.to_csv(out_file)


def plot_data(in_file, out_folder):
    yield_data = pd.read_csv(in_file)[['Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['Year', 'State', 'County', 'Value']
    if yield_data.Value.dtype != float:
        yield_data['Value'] = yield_data['Value'].str.replace(',', '')
        yield_data = yield_data.astype({'Year': int, 'State': int, 'County': int, 'Value': float})

    quantiles = {}
    for q in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]:
        quantiles[q] = yield_data.Value.quantile(q)
    quantiles[0.0] = yield_data.Value.min()
    quantiles[1.0] = yield_data.Value.max()
    print(quantiles)

    yield_per_year_dic = defaultdict(dict)

    for yd in yield_data.itertuples():
        year, state, county, value = yd.Year, yd.State, int(yd.County), yd.Value
        state = str(state).zfill(2)
        county = str(county).zfill(3)

        yield_per_year_dic[year][state + county] = value

    for year in np.unique(list(yield_per_year_dic.keys())):
        counties_plot(yield_per_year_dic[year], Path('{}/{}_yield.html'.format(out_folder, year)), quantiles)
    save_colorbar(out_folder, quantiles)


def get_counties_for_crop(county_file, crop_file, out_file):
    yield_data = pd.read_csv(crop_file)[['Year', 'State ANSI', 'County ANSI', 'Value']]
    counties = yield_data.drop_duplicates(subset=['State ANSI', 'County ANSI'])
    counties.columns = ['Year', 'State', 'County', 'Value']
    crop_counties = [int(str(int(yd.State)).zfill(2) + str(int(yd.County)).zfill(3)) for yd in counties.itertuples()]

    fh_in = Dataset(county_file, 'r')
    fh_out = Dataset(out_file, 'w')

    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        if v_name in ['lat', 'lon']:
            outVar[:] = varin[:]
        else:
            mask_values = np.in1d(varin[:], crop_counties).reshape(varin[:].shape)
            outVar[:] = ma.array(varin[:], mask=~mask_values)

    fh_in.close()
    fh_out.close()


def plot_counties(in_file):
    fh = Dataset(in_file, 'r')
    county_labels = fh.variables['county_label'][:]
    print(len(np.unique(county_labels.compressed())))
    fh.close()

    county_labels = np.unique(county_labels.compressed())
    county_labels = [[str(x).zfill(5)[:2], str(x).zfill(5)[2:]] for x in county_labels]

    data_dic = {}
    for state, county in county_labels:
        data_dic[state + county] = 100
    fake_quantiles = {x: 1 for x in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]}
    counties_plot(data_dic, Path('../../processed_data/crop_yield/{}.html'.format(in_file[:-3])), fake_quantiles)

    return data_dic.keys()


def get_counties_for_crops():
    # soybeans
    get_counties_for_crop('../../processed_data/counties/us_counties.nc',
                          '../../processed_data/crop_yield/soybeans_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/soybeans_us_counties.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/soybeans_us_counties.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro.nc',
                          '../../processed_data/crop_yield/soybeans_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/soybeans_us_counties_cro.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/soybeans_us_counties_cro.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro_cvm.nc',
                          '../../processed_data/crop_yield/soybeans_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/soybeans_us_counties_cro_cvm.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/soybeans_us_counties_cro_cvm.nc')

    # corn
    get_counties_for_crop('../../processed_data/counties/us_counties.nc',
                          '../../processed_data/crop_yield/corn_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/corn_us_counties.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/corn_us_counties.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro.nc',
                          '../../processed_data/crop_yield/corn_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/corn_us_counties_cro.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/corn_us_counties_cro.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro_cvm.nc',
                          '../../processed_data/crop_yield/corn_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/corn_us_counties_cro_cvm.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/corn_us_counties_cro_cvm.nc')

    # cotton
    get_counties_for_crop('../../processed_data/counties/us_counties.nc',
                          '../../processed_data/crop_yield/cotton_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/cotton_us_counties.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/cotton_us_counties.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro.nc',
                          '../../processed_data/crop_yield/cotton_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/cotton_us_counties_cro.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/cotton_us_counties_cro.nc')
    get_counties_for_crop('../../processed_data/counties/us_counties_cro_cvm.nc',
                          '../../processed_data/crop_yield/cotton_1999_2018.csv',
                          '../../processed_data/crop_yield/county_locations/cotton_us_counties_cro_cvm.nc')
    plot_counties('../../processed_data/crop_yield/county_locations/cotton_us_counties_cro_cvm.nc')


def analyze_patch_size(croptype):
    fh_us = Dataset('../../processed_data/crop_yield/{}_us_counties.nc'.format(croptype), 'r')
    fh_us_cro = Dataset('../../processed_data/crop_yield/{}_us_counties_cro.nc'.format(croptype), 'r')
    fh_us_cro_cvm = Dataset('../../processed_data/crop_yield/{}_us_counties_cro_cvm.nc'.format(croptype), 'r')

    us_counties, us_sizes = np.unique(fh_us.variables['county_label'][:].compressed(), return_counts=True)
    us_cro_counties, us_cro_sizes = np.unique(fh_us_cro.variables['county_label'][:].compressed(), return_counts=True)
    us_cro_cvm_counties, us_cro_cvm_sizes = np.unique(fh_us_cro_cvm.variables['county_label'][:].compressed(),
                                                      return_counts=True)
    us_dic = defaultdict(lambda: -1, {k: v for k, v in zip(us_counties, us_sizes)})
    us_cro_dic = defaultdict(lambda: -1, {k: v for k, v in zip(us_cro_counties, us_cro_sizes)})
    us_cro_cvm_dic = defaultdict(lambda: -1, {k: v for k, v in zip(us_cro_cvm_counties, us_cro_cvm_sizes)})

    for k in us_dic:
        print(k, us_dic[k], us_cro_dic[k], us_cro_cvm_dic[k])


def analyze_harvested_size(croptype):
    print(croptype)
    harvested_data = pd.read_csv('../../processed_data/crop_yield/harvested_areas/{}_1999_2018.csv'.format(croptype))
    important_columns = ['Year', 'State ANSI', 'County ANSI', 'Value']
    harvested_data = harvested_data.dropna(subset=important_columns, how='any')[['Year', 'State ANSI', 'County ANSI',
                                                                                 'Value']]
    harvested_data.columns = ['Year', 'State', 'County', 'Value']
    if harvested_data.Value.dtype != float:
        harvested_data['Value'] = harvested_data['Value'].str.replace(',', '')
        harvested_data = harvested_data.astype({'Year': int, 'State': int, 'County': int, 'Value': float})
        # convert from acres to square kilometers
        harvested_data['Value'] = harvested_data['Value'] / 247.105

    print(harvested_data.Value.describe(percentiles=[.1, .25, .5, .75, .9]))
    print(harvested_data[harvested_data.Value < 625].count().Year / len(harvested_data))


# County-level crop production is calculated by multiplying crop yield with harvest area.
def analyze_productions(croptype):
    print(croptype)
    yield_data = pd.read_csv('../../processed_data/crop_yield/{}_1999_2018.csv'.format(croptype))[['Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['Year', 'State', 'County', 'Value']
    if yield_data.Value.dtype != float:
        yield_data['Value'] = yield_data['Value'].str.replace(',', '')
        yield_data = yield_data.astype({'Year': int, 'State': int, 'County': int, 'Value': float})

    harvested_data = pd.read_csv('../../processed_data/crop_yield/harvested_areas/{}_1999_2018.csv'.format(croptype))
    important_columns = ['Year', 'State ANSI', 'County ANSI', 'Value']
    harvested_data = harvested_data.dropna(subset=important_columns, how='any')[['Year', 'State ANSI', 'County ANSI',
                                                                                 'Value']]
    harvested_data.columns = ['Year', 'State', 'County', 'Value']
    if harvested_data.Value.dtype != float:
        harvested_data['Value'] = harvested_data['Value'].str.replace(',', '')
        harvested_data = harvested_data.astype({'Year': int, 'State': int, 'County': int, 'Value': float})
        # convert from acres to square kilometers
        harvested_data['Value'] = harvested_data['Value']

    production_data = pd.DataFrame.merge(yield_data, harvested_data, on=['Year', 'State', 'County'], how='outer',
                                         suffixes=['_yield', '_harvest'])
    production_data['Production'] = production_data['Value_yield'] * production_data['Value_harvest']

    state_productions = production_data.groupby(['State'])['Production'].sum().to_dict()

    states, productions = zip(*sorted(state_productions.items(), key=itemgetter(1), reverse=True))
    total = sum(productions)
    i = 0
    for state, production, cum_perc in zip(states, productions, (100*subtotal/total for subtotal in np.cumsum(productions))):
        print(i, state, state_dic[state], production, cum_perc)
        i += 1


if __name__ == '__main__':
    # clean_data('../../processed_data/crop_yield/origi/soybeans_1999_2018.csv',
    #            '../../processed_data/crop_yield/soybeans_1999_2018.csv')
    # clean_data('../../processed_data/crop_yield/origi/soybeans_1999_2018.csv',
    #            '../../processed_data/crop_yield/soybeans_2000_2018.csv', dropyear=1999)
    # clean_data('../../processed_data/crop_yield/origi/grain_corn_1999_2018.csv',
    #            '../../processed_data/crop_yield/corn_1999_2018.csv')
    # clean_data('../../processed_data/crop_yield/origi/upland_cotton_1999_2018.csv',
    #            '../../processed_data/crop_yield/cotton_1999_2018.csv')
    # plot_data('../../processed_data/crop_yield/soybeans_1999_2018.csv',
    #           '../../processed_data/crop_yield/plots/soybeans')
    # plot_data('../../processed_data/crop_yield/corn_1999_2018.csv',
    #           '../../processed_data/crop_yield/plots/corn')
    # plot_data('../../processed_data/crop_yield/cotton_1999_2018.csv',
    #           '../../processed_data/crop_yield/plots/cotton')

    # get_counties_for_crops()

    # soybeans
    # analyze_patch_size('soybeans')

    # analyze_harvested_size('soybeans')
    # analyze_harvested_size('corn')
    # analyze_harvested_size('cotton')

    analyze_productions('soybeans')
    # analyze_productions('corn')
    # analyze_productions('cotton')
