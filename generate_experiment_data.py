#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from data_preprocessing.sample_quadruplets import generate_training_for_counties
from data_preprocessing.postprocess import mask_non_major_states
from data_preprocessing.postprocess import generate_no_spatial_for_counties
from data_preprocessing.postprocess import obtain_channel_wise_mean_std
from data_preprocessing.sample_quadruplets import generate_training_for_pretrained

if __name__ == '__main__':
    # MAJOR_STATES = [17, 18, 19, 20, 27, 29, 31, 38, 39, 46, 21, 55, 26]
    # ['Illinois', 'Indiana', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'Ohio',
    # 'South Dakota', 'Kentucky', 'Wisconsin', 'Michigan']
    mask_non_major_states('experiment_data/spatial_temporal/nc_files_unmasked',
                          'experiment_data/spatial_temporal/nc_files',
                          'processed_data/counties/lst/us_counties.nc',
                          MAJOR_STATES)

    generate_no_spatial_for_counties(yield_data_dir='processed_data/crop_yield',
                                     ppt_file='experiment_data/spatial_temporal/nc_files/2003.nc',
                                     county_location_file='processed_data/counties/lst/us_counties_cro_cvm_locations.csv',
                                     out_dir='experiment_data/no_spatial',
                                     img_dir='experiment_data/spatial_temporal/nc_files',
                                     croptype='soybeans',
                                     start_month=3,
                                     end_month=9,
                                     start_index=1)

    obtain_channel_wise_mean_std('experiment_data/spatial_temporal/nc_files')

    for nr in [5, 50, 100, 1000, None]:
        generate_training_for_counties(out_dir='experiment_data/spatial_temporal/counties',
                                       img_dir='experiment_data/spatial_temporal/nc_files',
                                       start_month=3, end_month=9, start_month_index=1, n_spatial_neighbor=1, n_distant=1,
                                       img_timestep_quadruplets=
                                       'experiment_data/spatial_temporal/counties/img_timestep_quadruplets_hard.csv',
                                       img_size=50, neighborhood_radius=nr, distant_radius=None, prenorm=True)

    generate_training_for_pretrained(out_dir='experiment_data/spatial_temporal/counties',
                                     img_dir='experiment_data/spatial_temporal/nc_files',
                                     n_quadruplets=100000,
                                     start_year=2003, end_year=2012, start_month=3, end_month=9, start_month_index=1,
                                     n_spatial_neighbor=1, n_distant=1,
                                     img_size=50, neighborhood_radius=10, distant_radius=50, prenorm=True)
    generate_training_for_pretrained(out_dir='experiment_data/spatial_temporal/counties',
                                     img_dir='experiment_data/spatial_temporal/nc_files',
                                     n_quadruplets=100000,
                                     start_year=2003, end_year=2012, start_month=3, end_month=9, start_month_index=1,
                                     n_spatial_neighbor=1, n_distant=1,
                                     img_size=50, neighborhood_radius=25, distant_radius=100, prenorm=True)
