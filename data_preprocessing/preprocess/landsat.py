#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def combine_landsat():
    fh_out = Dataset('../../processed_data/landsat/20180719.nc', 'w')

    flag = False
    for i in range(1, 8):
        fh_in = Dataset('../../raw_data/landsat/nebraska/SRB{}_20180719.nc'.format(i), 'r')
        if not flag:
            lats, lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]

            fh_out.createDimension("lat", len(lats))
            fh_out.createDimension("lon", len(lons))

            for v_name, varin in fh_in.variables.items():
                if v_name in ["lat", "lon"]:
                    outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            fh_out.variables["lat"][:] = lats[:]
            fh_out.variables["lon"][:] = lons[:]
            flag = True

        for v_name, varin in fh_in.variables.items():
            if v_name == 'Band1':
                outVar = fh_out.createVariable('band{}'.format(i), varin.datatype, ('lat', 'lon'))
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = ma.masked_less(varin[:], 0)

        fh_in.close()
    fh_out.close()


# 20190604
def subset_landsat(lat1, lat2, lon1, lon2):
    fh_out = Dataset('../../processed_data/landsat/2019155.nc', 'w')

    flag = False
    lat_indices, lon_indices = None, None
    for i in range(1, 8):
        fh_in = Dataset('../../raw_data/landsat/SRB{}_doy2019155.nc'.format(i), 'r')
        if not flag:
            lats, lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]
            lat_indices = np.searchsorted(lats, [lat2, lat1])
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
            flag = True

        for v_name, varin in fh_in.variables.items():
            if v_name == 'Band1':
                outVar = fh_out.createVariable('band{}'.format(i), varin.datatype, ('lat', 'lon'))
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = ma.masked_less(varin[lat_indices[0]: lat_indices[1], lon_indices[0]: lon_indices[1]], 0)

        fh_in.close()
    fh_out.close()


if __name__ == '__main__':
    # combine_landsat()
    subset_landsat(41.2047, 40.0268, -122.0304, -120.4676)
