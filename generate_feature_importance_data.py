#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crop_yield_prediction import CLIMATE_VARS

import os
import numpy as np


def generate_feature_importance_data_exclude(in_dir, out_dir, exclude_group):
    os.makedirs(out_dir, exist_ok=True)

    include_indices = [i for i, x in enumerate(CLIMATE_VARS) if x not in exclude_group]
    print(exclude_group, include_indices)
    for f in os.listdir(in_dir):
        if f.endswith('.npy'):
            in_data = np.load('{}/{}'.format(in_dir, f))
            out_data = in_data[:, :, :, include_indices, :, :]
            # print(out_data.shape)
            np.save('{}/{}'.format(out_dir, f), out_data)


if __name__ == '__main__':
    # exclude_groups = [('ppt',), ('evi', 'ndvi'), ('elevation',), ('lst_day', 'lst_night'),
    #                   ('clay', 'sand', 'silt')]
    exclude_groups = [('ppt', 'elevation', 'lst_day', 'lst_night', 'clay', 'sand', 'silt')]
    for eg in exclude_groups:
        generate_feature_importance_data_exclude(in_dir='data/spatial_temporal/counties/nr_25_dr100',
                                                 out_dir='data/spatial_temporal/counties/nr_25_dr100_{}'.format('_'.join(eg)),
                                                 exclude_group=eg)
