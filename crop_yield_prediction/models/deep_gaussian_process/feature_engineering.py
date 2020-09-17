#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapt code from https://github.com/gabrieltseng/pycrop-yield-prediction


from crop_yield_prediction import CLIMATE_VARS

import pandas as pd
import numpy as np

MAX_BIN_VAL = {'ppt': 179.812, 'evi': 0.631, 'ndvi': 0.850, 'elevation': 961.420, 'lst_day': 309.100,
               'lst_night': 293.400, 'clay': 47.0, 'sand': 91.0, 'silt': 70.0}
MIN_BIN_VAL = {'ppt': 11.045, 'evi': 0.084, 'ndvi': 0.138, 'elevation': 175.0, 'lst_day': 269.640,
               'lst_night': 261.340, 'clay': 4.0, 'sand': 13.0, 'silt': 10.0}


def _calculate_histogram(image, num_bins=32):
    """
    Input image shape: (n_variables, n_timesteps, 50, 50)
    """
    hist = []
    n_variables, n_timesteps = image.shape[:2]
    for var_idx in range(n_variables):
        bin_seq = np.linspace(MIN_BIN_VAL[CLIMATE_VARS[var_idx]], MAX_BIN_VAL[CLIMATE_VARS[var_idx]], num_bins + 1)
        im = image[var_idx]
        imhist = []
        for ts_idx in range(n_timesteps):
            density, _ = np.histogram(im[ts_idx, :, :], bin_seq, density=False)
            # max() prevents divide by 0
            imhist.append(density / max(1, density.sum()))
        hist.append(np.stack(imhist))

    # [bands, times, bins]
    hist = np.stack(hist)

    return hist


def get_features_for_deep_gaussian():
    output_images = []
    yields = []
    years = []
    locations = []
    state_county_info = []

    yield_data = pd.read_csv('data/deep_gaussian/deep_gaussian_dim_y.csv')[['state', 'county', 'year', 'value', 'lat', 'lon']]
    yield_data.columns = ['state', 'county', 'year', 'value', 'lat', 'lon']

    for idx, yield_data in enumerate(yield_data.itertuples()):
        year, county, state = yield_data.year, yield_data.county, yield_data.state

        # [1, n_timesteps, 1+n_temporal_neighbor+n_spatial_neighbor+n_distant, n_variables, 50, 50]
        image = np.load('data/deep_gaussian/nr_25/{}.npy'.format(idx))

        # get anchor image from shape (1, n_timesteps, 4, n_variables, 50, 50)
        # to shape (n_timestep, n_variables, 50, 50)
        image = image[0, :, 0, :, :, :]
        # shape (n_variables, n_timesteps, 50, 50)
        image = np.swapaxes(image, 0, 1)

        image = _calculate_histogram(image, num_bins=32)

        output_images.append(image)
        yields.append(yield_data.value)
        years.append(year)

        lat, lon = yield_data.lat, yield_data.lon
        locations.append(np.array([lon, lat]))

        state_county_info.append(np.array([int(state), int(county)]))

        # print(f'County: {int(county)}, State: {state}, Year: {year}, Output shape: {image.shape}')

    np.savez('data/deep_gaussian/data.npz',
             output_image=np.stack(output_images), output_yield=np.array(yields),
             output_year=np.array(years), output_locations=np.stack(locations),
             output_index=np.stack(state_county_info))
