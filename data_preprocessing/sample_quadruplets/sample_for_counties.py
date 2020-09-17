#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("..")

from data_preprocessing import CLIMATE_VARS


def generate_dims_for_counties(croptype):
    yield_data = pd.read_csv('../../processed_data/crop_yield/{}_2000_2018.csv'.format(croptype))[[
        'Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['year', 'state', 'county', 'value']
    ppt_fh = Dataset('../../experiment_data/spatial_temporal/nc_files/2014.nc', 'r')
    v_ppt = ppt_fh.variables['ppt'][0, :, :]
    if yield_data.value.dtype != float:
        yield_data['value'] = yield_data['value'].str.replace(',', '')
    yield_data = yield_data.astype({'year': int, 'state': int, 'county': int, 'value': float})
    counties = pd.read_csv('../../processed_data/counties/lst/us_counties_cro_cvm_locations.csv')
    county_dic = {}
    for c in counties.itertuples():
        state, county, lat0, lat1, lon0, lon1 = c.state, c.county, c.lat0, c.lat1, c.lon0, c.lon1
        county_dic[(state, county)] = [lat0, lat1, lon0, lon1]

    yield_dim_csv = []
    for yd in yield_data.itertuples():
        year, state, county, value = yd.year, yd.state, yd.county, yd.value

        if (state, county) not in county_dic:
            continue

        lat0, lat1, lon0, lon1 = county_dic[(state, county)]
        assert lat1 - lat0 == 49
        assert lon1 - lon0 == 49

        selected_ppt = v_ppt[lat0:lat1+1, lon0:lon1+1]
        if ma.count_masked(selected_ppt) != 0:
            continue

        yield_dim_csv.append([state, county, year, value, (lat0+lat1)//2+1, (lon0+lon1)//2+1])

    yield_dim_csv = pd.DataFrame(yield_dim_csv, columns=['state', 'county', 'year', 'value', 'lat', 'lon'])
    yield_dim_csv.to_csv('../../experiment_data/spatial_temporal/counties/dim_y.csv')


def get_imgs_and_timesteps_for_quadruplets(yield_dim_csv, start_month, end_month, n_distant, sample_case):
    yield_dim_data = pd.read_csv(yield_dim_csv)

    img_timestep_quadruplets = []
    for i_data, data in enumerate(yield_dim_data.itertuples()):
        a_year, a_lat, a_lon = data.year, data.lat, data.lon

        exp = [i_data, a_year, a_lat, a_lon]
        months_head = []
        for a_month in range(start_month, end_month+1):
            for i_distant in range(n_distant):
                if sample_case == 'hard':
                    exp += [a_year, a_month]
                elif sample_case == 'soft':
                    i_distant_year = np.random.randint(2000, a_year+1)
                    i_distant_month = np.random.randint(start_month, end_month+1)
                    exp += [i_distant_year, i_distant_month]
                months_head.append('a_month{}_d{}_year'.format(a_month, i_distant))
                months_head.append('a_month{}_d{}_month'.format(a_month, i_distant))
        img_timestep_quadruplets.append(exp)

    img_timestep_quadruplets = pd.DataFrame(img_timestep_quadruplets,
                                         columns=['index', 'a_year', 'a_lat', 'a_lon'] + months_head)
    img_timestep_quadruplets.to_csv('../../experiment_data/spatial_temporal/counties/img_timestep_quadruplets_{}.csv'.format(sample_case), index=False)


# output dimension: [n_samples, n_timesteps, 1+n_temporal_neighbor+n_spatial_neighbor+n_distant, n_variables, 50, 50]
def generate_training_for_counties(out_dir, img_dir, start_month, end_month, start_month_index, n_spatial_neighbor, n_distant,
                                   img_timestep_quadruplets, img_size, neighborhood_radius, distant_radius=None, prenorm=True):
    if distant_radius is None:
        output_dir = '{}/nr_{}'.format(out_dir, neighborhood_radius)
    else:
        output_dir = '{}/nr_{}_dr{}'.format(out_dir, neighborhood_radius, distant_radius)
    os.makedirs(output_dir, exist_ok=True)

    size_even = (img_size % 2 == 0)
    tile_radius = img_size // 2

    img_timestep_quadruplets = pd.read_csv(img_timestep_quadruplets)
    columns = ['index', 'a_year', 'a_lat', 'a_lon']
    for a_month in range(start_month, end_month + 1):
        for i_distant in range(n_distant):
            columns.append('a_month{}_d{}_year'.format(a_month, i_distant))
            columns.append('a_month{}_d{}_month'.format(a_month, i_distant))
    column_index = {x: i+1 for i, x in enumerate(columns)}
    print(column_index)

    # monthly mean
    # {0: [57.3017, 0.15911582, 0.30263194, 349.417, 277.6782, 268.29166, 19.372774, 38.962997, 48.396523],
    #  1: [73.980095, 0.19241332, 0.35961938, 349.417, 286.09885, 273.22183, 19.372774, 38.962997, 48.396523],
    #  2: [87.33122, 0.27037004, 0.46616226, 349.417, 294.85776, 279.05136, 19.372774, 38.962997, 48.396523],
    #  3: [106.66116, 0.38423842, 0.5934064, 349.417, 299.4103, 284.4472, 19.372774, 38.962997, 48.396523],
    #  4: [111.04675, 0.46401384, 0.6796355, 349.417, 302.36234, 289.90076, 19.372774, 38.962997, 48.396523],
    #  5: [100.82861, 0.5001915, 0.7197062, 349.417, 303.2484, 292.21436, 19.372774, 38.962997, 48.396523],
    #  6: [93.255714, 0.4844686, 0.71926653, 349.417, 302.26636, 291.2553, 19.372774, 38.962997, 48.396523],
    #  7: [88.390526, 0.41577676, 0.67133075, 349.417, 299.28165, 287.00778, 19.372774, 38.962997, 48.396523]}
    # monthly std
    # {0: [49.994095, 0.09068172, 0.18281896, 258.4355, 9.178257, 8.026086, 17.579718, 24.665548, 20.690763],
    #  1: [56.513268, 0.084073044, 0.15483402, 258.4355, 7.8059173, 6.699706, 17.579718, 24.665548, 20.690763],
    #  2: [53.212543, 0.11533181, 0.17148177, 258.4355, 5.039537, 5.1716127, 17.579718, 24.665548, 20.690763],
    #  3: [60.39661, 0.1439103, 0.18301234, 258.4355, 4.484442, 4.53816, 17.579718, 24.665548, 20.690763],
    #  4: [60.862434, 0.13719948, 0.16091526, 258.4355, 4.6158304, 3.6706781, 17.579718, 24.665548, 20.690763],
    #  5: [58.666737, 0.13492998, 0.15656078, 258.4355, 5.140572, 3.0179217, 17.579718, 24.665548, 20.690763],
    #  6: [60.55039, 0.14212538, 0.16778886, 258.4355, 4.962786, 3.2834055, 17.579718, 24.665548, 20.690763],
    #  7: [64.83031, 0.12455596, 0.17052796, 258.4355, 4.5033474, 3.5745926, 17.579718, 24.665548, 20.690763]}
    mean_file = open('{}/monthly_channel_wise_mean.pkl'.format(img_dir), 'rb')
    std_file = open('{}/monthly_channel_wise_std.pkl'.format(img_dir), 'rb')
    monthly_mean = pickle.load(mean_file)
    monthly_std = pickle.load(std_file)

    img_dic = {}
    for year in range(2000, 2019):
        fh = Dataset('{}/{}.nc'.format(img_dir, year))
        img = []
        for cv in CLIMATE_VARS:
            img.append(fh.variables[cv][:])
        img = ma.asarray(img)
        # (n_variables, n_timesteps, n_lat, n_lon)
        img_shape = img.shape
        fh.close()
        img_dic[year] = img

    n_quadruplets = img_timestep_quadruplets['index'].max() + 1
    print('Number of quadruplets: {}'.format(n_quadruplets))
    n_t = end_month - start_month + 1
    n_tiles_per_file = 1
    n_tiles = 0
    tiles = []
    n_samples = 0
    a_lats, a_lons, sn_lats, sn_lons, d_lats, d_lons = [], [], [], [], [], []
    print('Start sampling...')
    for data in img_timestep_quadruplets.itertuples():
        index, a_year, a_lat, a_lon = data.index, data.a_year, data.a_lat, data.a_lon
        quadruplets_tile = np.zeros((n_t, (1+1+n_spatial_neighbor+n_distant), len(CLIMATE_VARS), img_size, img_size))

        for a_month in range(start_month, end_month+1):
            a_month_tile_index = a_month - start_month

            current_ts_index = a_month-start_month+start_month_index
            lat0, lat1, lon0, lon1 = _get_lat_lon_range(a_lat, a_lon, tile_radius, size_even)
            current_tile = img_dic[a_year][:, current_ts_index, lat0:lat1+1, lon0:lon1+1]
            assert ma.count_masked(current_tile) == 0
            current_tile = np.asarray(current_tile)
            if prenorm:
                current_tile = _prenormalize_tile(current_tile, monthly_mean[current_ts_index], monthly_std[current_ts_index])
            quadruplets_tile[a_month_tile_index, 0] = current_tile
            n_samples += 1
            a_lats.append(a_lat)
            a_lons.append(a_lon)

            tn_tile = img_dic[a_year][:, current_ts_index-1, lat0:lat1 + 1, lon0:lon1 + 1]
            assert ma.count_masked(tn_tile) == 0
            tn_tile = np.asarray(tn_tile)
            if prenorm:
                tn_tile = _prenormalize_tile(tn_tile, monthly_mean[current_ts_index-1], monthly_std[current_ts_index-1])
            quadruplets_tile[a_month_tile_index, 1] = tn_tile
            n_samples += 1

            for i_spatial_neighbor in range(n_spatial_neighbor):
                sn_tile, sn_lat, sn_lon = _sample_neighbor(img_dic[a_year], a_lat, a_lon, neighborhood_radius,
                                                           tile_radius, current_ts_index, size_even)
                assert ma.count_masked(sn_tile) == 0
                sn_tile = np.asarray(sn_tile)
                if prenorm:
                    sn_tile = _prenormalize_tile(sn_tile, monthly_mean[current_ts_index], monthly_std[current_ts_index])
                quadruplets_tile[a_month_tile_index, 2+i_spatial_neighbor] = sn_tile
                n_samples += 1
                sn_lats.append(sn_lat)
                sn_lons.append(sn_lon)

            for i_distant in range(n_distant):
                d_year = data[column_index['a_month{}_d{}_year'.format(a_month, i_distant)]]
                d_month = data[column_index['a_month{}_d{}_month'.format(a_month, i_distant)]]
                if d_year == a_year and d_month == a_month:
                    d_tile, d_lat, d_lon = _sample_distant_same(img_dic[d_year], a_lat, a_lon, neighborhood_radius,
                                                                distant_radius,
                                                                tile_radius, current_ts_index, size_even)
                    assert ma.count_masked(d_tile) == 0
                    d_tile = np.asarray(d_tile)
                    if prenorm:
                        d_tile = _prenormalize_tile(d_tile, monthly_mean[current_ts_index], monthly_std[current_ts_index])
                    quadruplets_tile[a_month_tile_index, 2+n_spatial_neighbor+i_distant] = d_tile
                    n_samples += 1
                    d_lats.append(d_lat)
                    d_lons.append(d_lon)
                else:
                    print('Wrong sampling!')
                    d_ts_index = d_month - start_month + start_month_index
                    d_tile, d_lat, d_lon = _sample_distant_diff(img_dic[d_year], tile_radius, d_ts_index, size_even)
                    assert ma.count_masked(d_tile) == 0
                    d_tile = np.asarray(d_tile)
                    if prenorm:
                        d_tile = _prenormalize_tile(d_tile, monthly_mean[d_ts_index], monthly_std[d_ts_index])
                    quadruplets_tile[a_month_tile_index, 2 + n_spatial_neighbor + i_distant] = d_tile
                    n_samples += 1
                    d_lats.append(d_lat)
                    d_lons.append(d_lon)

        # output dimension: [n_samples, n_timesteps, 1+n_temporal_neighbor+n_spatial_neighbor+n_distant, n_variables, 50, 50]
        tiles.append(quadruplets_tile)
        if len(tiles) == n_tiles_per_file or (n_tiles + len(tiles)) == n_quadruplets:
            if n_tiles_per_file > 1:
                np.save('{}/{}_{}.npy'.format(output_dir, n_tiles, n_tiles + len(tiles) - 1), np.asarray(tiles, dtype=np.float32))
            else:
                np.save('{}/{}.npy'.format(output_dir, n_tiles), np.asarray(tiles, dtype=np.float32))
            assert n_samples == len(tiles) * n_t * (1 + 1 + n_spatial_neighbor + n_distant), n_samples
            n_tiles += len(tiles)
            tiles = []
            n_samples = 0

    plot_sampled_centers(a_lats, a_lons, img_shape, output_dir, 'a_dims')
    plot_sampled_centers(sn_lats, sn_lons, img_shape, output_dir, 'sn_dims')
    plot_sampled_centers(d_lats, d_lons, img_shape, output_dir, 'd_dims')


def _prenormalize_tile(tile, means, stds):
    means = np.asarray(means).reshape((-1, 1, 1))
    stds = np.asarray(stds).reshape((-1, 1, 1))
    return (tile - means) / stds


def _get_lat_lon_range(a_lat, a_lon, tile_radius, size_even):
    lat0, lon0 = a_lat - tile_radius, a_lon - tile_radius
    lat1 = a_lat + tile_radius - 1 if size_even else a_lat + tile_radius
    lon1 = a_lon + tile_radius - 1 if size_even else a_lon + tile_radius

    return lat0, lat1, lon0, lon1


def _sample_neighbor(img, a_lat, a_lon, neighborhood_radius, tile_radius, timestep, size_even):
    if neighborhood_radius is None:
        return _sample_distant_diff(img, tile_radius, timestep, size_even)

    _, _, img_h, img_w = img.shape
    while True:
        n_lat, n_lon = a_lat, a_lon
        while n_lat == a_lat and n_lon == a_lon:
            n_lat = np.random.randint(max(a_lat - neighborhood_radius, tile_radius),
                                      min(a_lat + neighborhood_radius, img_h - tile_radius))
            n_lon = np.random.randint(max(a_lon - neighborhood_radius, tile_radius),
                                      min(a_lon + neighborhood_radius, img_w - tile_radius))
        lat0, lat1, lon0, lon1 = _get_lat_lon_range(n_lat, n_lon, tile_radius, size_even)
        tile = img[:, timestep, lat0:lat1+1, lon0:lon1+1]
        if ma.count_masked(tile) == 0:
            break

    return tile, n_lat, n_lon


def _sample_distant_same(img, a_lat, a_lon, neighborhood_radius, distant_radius, tile_radius, timestep, size_even):
    if neighborhood_radius is None:
        return _sample_distant_diff(img, tile_radius, timestep, size_even)

    _, _, img_h, img_w = img.shape
    while True:
        d_lat, d_lon = a_lat, a_lon

        if distant_radius is None:
            while (d_lat >= a_lat - neighborhood_radius) and (d_lat <= a_lat + neighborhood_radius):
                d_lat = np.random.randint(tile_radius, img_h - tile_radius)
            while (d_lon >= a_lon - neighborhood_radius) and (d_lon <= a_lon + neighborhood_radius):
                d_lon = np.random.randint(tile_radius, img_w - tile_radius)
        else:
            while ((d_lat >= a_lat - neighborhood_radius) and (d_lat <= a_lat + neighborhood_radius)) \
                    or d_lat >= a_lat + distant_radius \
                    or d_lat <= a_lat - distant_radius:
                d_lat = np.random.randint(tile_radius, img_h - tile_radius)
            while ((d_lon >= a_lon - neighborhood_radius) and (d_lon <= a_lon + neighborhood_radius))\
                    or d_lon >= a_lon + distant_radius \
                    or d_lon <= a_lon - distant_radius:
                d_lon = np.random.randint(tile_radius, img_w - tile_radius)
        lat0, lat1, lon0, lon1 = _get_lat_lon_range(d_lat, d_lon, tile_radius, size_even)
        tile = img[:, timestep, lat0:lat1 + 1, lon0:lon1 + 1]
        if ma.count_masked(tile) == 0:
            break

    return tile, d_lat, d_lon


def _sample_distant_diff(img, tile_radius, timestep, size_even):
    _, _, img_h, img_w = img.shape
    while True:
        d_lat = np.random.randint(tile_radius, img_h - tile_radius)
        d_lon = np.random.randint(tile_radius, img_w - tile_radius)
        lat0, lat1, lon0, lon1 = _get_lat_lon_range(d_lat, d_lon, tile_radius, size_even)
        tile = img[:, timestep, lat0:lat1 + 1, lon0:lon1 + 1]
        if ma.count_masked(tile) == 0:
            break

    return tile, d_lat, d_lon


def plot_sampled_centers(lats, lons, img_shape, out_dir, name):
    c, t, h, w = img_shape

    plt.scatter(lons, lats, s=5)
    plt.axis([0, w, 0, h])
    plt.savefig('{}/{}.jpg'.format(out_dir, name))
    plt.close()


if __name__ == '__main__':
    generate_dims_for_counties(croptype='soybeans')
    get_imgs_and_timesteps_for_quadruplets('../../experiment_data/spatial_temporal/counties/dim_y.csv', 3, 9, 1,
                                           sample_case='hard')
    get_imgs_and_timesteps_for_quadruplets('../../experiment_data/spatial_temporal/counties/dim_y.csv', 3, 9, 1,
                                           sample_case='soft')

    generate_training_for_counties(out_dir='../../experiment_data/spatial_temporal/counties',
                                   img_dir='../../experiment_data/spatial_temporal/nc_files',
                                   start_month=3, end_month=9, start_month_index=1, n_spatial_neighbor=1, n_distant=1,
                                   img_timestep_quadruplets=
                                   '../../experiment_data/spatial_temporal/counties/img_timestep_quadruplets.csv',
                                   img_size=50, neighborhood_radius=100)
