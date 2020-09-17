#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from data_preprocessing.utils import get_lat_lon_bins
from data_preprocessing import cdl_values_to_crops, crops_to_cdl_values
from data_preprocessing.utils import timeit

import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset


# water: Water Wetlands Aquaculture Open Water Perennial Ice/Snow
# urban: Developed Developed/Open Space Developed/Low Intensity Developed/Med Intensity Developed/High Intensity
# native: Clover/Wildflowers Forest Shrubland1 Deciduous Forest Evergreen Forest Mixed Forest Shrubland2 Woody Wetlands
# Herbaceous Wetlands
# idle/fallow: Sod/Grass Seed Fallow/Idle Cropland
# hay/pasture: Other Hay/Non Alfalfa Switchgrass Grassland/Pasture
# barren/missing: Barren1 Clouds/No Data Nonag/Undefined Barren2
ignored_labels = {"water": [83, 87, 92, 111, 112],
                  "urban": [82, 121, 122, 123, 124],
                  "native": [58, 63, 64, 141, 142, 143, 152, 190, 195],
                  "idle/fallow": [59, 61],
                  "hay/pasture": [37, 60, 176],
                  "barren/missing": [65, 81, 88, 131]}


def cdl_upscale(in_dir, in_file, out_dir, out_file, reso='40km', ignore=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ignored_lis = [x for lis in ignored_labels.values() for x in lis]
    kept_lis = [x for x in cdl_values_to_crops.keys() if x not in ignored_lis]

    # increasing
    lats = np.load('../../processed_data/prism/latlon/lat_{}.npy'.format(reso))
    lons = np.load('../../processed_data/prism/latlon/lon_{}.npy'.format(reso))
    _, _, lat_bins, lon_bins = get_lat_lon_bins(lats, lons)

    fh_in = Dataset(os.path.join(in_dir, in_file), 'r')
    fh_out = Dataset(os.path.join(out_dir, out_file), 'w')

    dic_var = {}
    for var in ['lat', 'lon']:
        dic_var[var] = fh_in.variables[var]
    # increasing
    dic_var['lat_value'] = dic_var['lat'][:]
    dic_var['lon_value'] = dic_var['lon'][:]

    fh_out.createDimension('lat', len(lats))
    fh_out.createDimension('lon', len(lons))

    for var in ['lat', 'lon']:
        outVar = fh_out.createVariable(var, 'f4', (var,))
        outVar.setncatts({k: dic_var[var].getncattr(k) for k in dic_var[var].ncattrs()})
        outVar[:] = lats if var == "lat" else lons

    cdl_value = fh_in.variables['Band1'][:]
    cdl_resampled_dic = {}
    for v in cdl_values_to_crops.values():
        if (ignore and crops_to_cdl_values[v] in kept_lis) or not ignore:
            cdl_resampled_dic[v] = np.full((len(lats), len(lons)), -1.0)

    for s in ["1", "2", "3"]:
        cdl_resampled_dic["cdl_" + s] = np.full((len(lats), len(lons)), -1.0)
        cdl_resampled_dic["cdl_fraction_" + s] = np.full((len(lats), len(lons)), -1.0)

    for id_lats in range(len(lats)):
        for id_lons in range(len(lons)):
            lats_index = np.searchsorted(dic_var['lat_value'],
                                         [lat_bins[id_lats], lat_bins[id_lats + 1]])
            lons_index = np.searchsorted(dic_var['lon_value'],
                                         [lon_bins[id_lons], lon_bins[id_lons + 1]])

            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                selected = cdl_value[np.array(range(lats_index[0], lats_index[1]))[:, None],
                                     np.array(range(lons_index[0], lons_index[1]))]
                # selected_size = selected.shape[0] * selected.shape[1]
                selected_compressed = selected.compressed()
                selected_size = len(selected_compressed)
                cdl_id, cdl_count = np.unique(selected_compressed, return_counts=True)

                # filter ignored_label after selected_size has been calculated
                if ignore:
                    new_cdl_id, new_cdl_count = [], []
                    for i, c in zip(cdl_id, cdl_count):
                        if i in kept_lis:
                            new_cdl_id.append(i)
                            new_cdl_count.append(c)
                    cdl_id, cdl_count = np.asarray(new_cdl_id), np.asarray(new_cdl_count)

                for i, c in zip(cdl_id, cdl_count):
                    cdl_resampled_dic[cdl_values_to_crops[i]][id_lats, id_lons] = c / selected_size
                cdl_count_sort_ind = np.argsort(-cdl_count)
                for i in range(3):
                    if len(cdl_id) > i:
                        cdl_resampled_dic["cdl_" + str(i+1)][id_lats, id_lons] = \
                            cdl_id[cdl_count_sort_ind[i]]
                        cdl_resampled_dic["cdl_fraction_" + str(i+1)][id_lats, id_lons] = \
                            cdl_count[cdl_count_sort_ind[i]] / selected_size
                    else:
                        cdl_resampled_dic["cdl_" + str(i + 1)][id_lats, id_lons] = -1
                        cdl_resampled_dic["cdl_fraction_" + str(i + 1)][id_lats, id_lons] = -1

    for v in cdl_values_to_crops.values():
        if (ignore and crops_to_cdl_values[v] in kept_lis) or not ignore:
            outVar = fh_out.createVariable("cdl_" + v.lower().replace(' ', '_').replace(' & ', '_').replace('/', '_'),
                                           'f4', ('lat', 'lon',))
            outVar[:] = cdl_resampled_dic[v][:]
            outVar[:] = ma.masked_equal(outVar, -1.0)
    for s in ["1", "2", "3"]:
        for t in ["cdl_", "cdl_fraction_"]:
            outVar = fh_out.createVariable(t + s, 'f4', ('lat', 'lon',))
            outVar[:] = cdl_resampled_dic[t + s][:]
            outVar[:] = ma.masked_equal(outVar, -1.0)

    fh_in.close()
    fh_out.close()


def upscaled_cdl_postprocess(in_file, out_dir, out_file, threshold=0.0):
    fh_in = Dataset(in_file, 'r')

    cdl_fraction_1 = fh_in.variables['cdl_fraction_1'][:]
    kept_cdls = ma.masked_where(cdl_fraction_1 < threshold, fh_in.variables['cdl_1'][:])
    cdl_id, cdl_count = np.unique(kept_cdls.compressed(), return_counts=True)
    for i, c in zip(cdl_id, cdl_count):
        print(cdl_values_to_crops[int(i)], c)


if __name__ == "__main__":
    timeit()
    # cdl_upscale('../../raw_data/cdl/2018_30m_cdls', '2018_30m_cdls.nc',
    #             '../../processed_data/cdl/40km', '2018_40km_cdls_crop_only.nc', reso='40km', ignore=True)
    upscaled_cdl_postprocess('../../processed_data/cdl/40km/2018_40km_cdls_crop_only.nc',
                             '', '')
    # print([x for lis in ignored_labels.values() for x in lis])
