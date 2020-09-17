#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import numpy as np
import sys
sys.path.append("..")

from data_preprocessing.utils import get_closet_date


def subset(in_file, out_file, lat1, lat2, lon1, lon2):
    fh_in = Dataset(in_file, 'r')
    fh_out = Dataset(out_file, 'w')

    lats, lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]
    lat_indices = lats.size - np.searchsorted(lats[::-1], [lat1, lat2], side="right")
    lon_indices = np.searchsorted(lons, [lon1, lon2])
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    for v_name, varin in fh_in.variables.items():
        if v_name in ["lat", "lon"]:
            outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
    fh_out.variables["lat"][:] = lats[:]
    fh_out.variables["lon"][:] = lons[:]

    for v_name, varin in fh_in.variables.items():
        if v_name not in ["lat", "lon"]:
            outVar = fh_out.createVariable(v_name, varin.datatype, ('lat', 'lon'))
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[lat_indices[0]: lat_indices[1], lon_indices[0]: lon_indices[1]]

    fh_in.close()
    fh_out.close()


if __name__ == '__main__':
    # subset('../../processed_data/nws_precip/500m/20190604.nc',
    #        '../../processed_data/local/ca_20190604/nws_precip.nc',
    #        41.2047, 40.0268, -122.0304, -120.4676)
    # subset('../../processed_data/elevation/500m.nc',
    #        '../../processed_data/local/ca_20190604/elevation.nc',
    #        41.2047, 40.0268, -122.0304, -120.4676)
    # subset('../../processed_data/soil_fraction/soil_fraction_usa_500m.nc',
    #        '../../processed_data/local/ca_20190604/soil_fraction.nc',
    #        41.2047, 40.0268, -122.0304, -120.4676)

    lai_date = get_closet_date('20190604', '../../processed_data/lai/500m')
    print(lai_date)
    subset('../../processed_data/lai/500m/{}.nc'.format(lai_date),
           '../../processed_data/local/ca_20190604/lai.nc',
           41.2047, 40.0268, -122.0304, -120.4676)
    lst_date = get_closet_date('20190604', '../../processed_data/lst/500m')
    print(lst_date)
    subset('../../processed_data/lst/500m/{}.nc'.format(lst_date),
           '../../processed_data/local/ca_20190604/lst.nc',
           41.2047, 40.0268, -122.0304, -120.4676)
    # subset('../../processed_data/soil_moisture/9km_500m/20190604.nc',
    #        '../../processed_data/local/ca_20190604/soil_moisture.nc',
    #        41.2047, 40.0268, -122.0304, -120.4676)

