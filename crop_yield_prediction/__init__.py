#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ['CLIMATE_VARS', 'STATIC_CLIMATE_VARS', 'DYNAMIC_CLIMATE_VARS']

CLIMATE_VARS = ['ppt', 'evi', 'ndvi', 'elevation', 'lst_day', 'lst_night', 'clay', 'sand', 'silt']
STATIC_CLIMATE_VARS = ['elevation', 'clay', 'sand', 'silt']
DYNAMIC_CLIMATE_VARS = [x for x in CLIMATE_VARS if x not in STATIC_CLIMATE_VARS]
