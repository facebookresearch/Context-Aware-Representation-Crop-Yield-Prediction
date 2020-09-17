#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset

import os
from osgeo import gdal, osr
import numpy as np
from pyproj import Proj, transform
import numpy.ma as ma

# gdalwarp -t_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' 2018_30m_cdls.img 2018_30m_cdls.tif
# gdal_translate -of netCDF PRISM_ppt_stable_4kmM3_201806_bil.bil PRISM_ppt_stable_4kmM3_201806.nc


def cdl_convert_to_nc(in_dir, in_file, out_dir, out_file):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raster = gdal.Open(os.path.join(in_dir, in_file))
    cdl_values = raster.ReadAsArray()
    geo = raster.GetGeoTransform()
    projWKT = raster.GetProjection()
    proj = osr.SpatialReference()
    proj.ImportFromWkt(projWKT)
    # n_lat, n_lon = np.shape(cdl_values)
    # b = raster.GetGeoTransform()
    # lons = (np.arange(n_lon) * b[1] + b[0])
    # lats = (np.arange(n_lat) * b[5] + b[3])
    #
    # fh_out = Dataset(os.path.join(out_dir, out_file), "w")
    # fh_out.createDimension("lat", len(lats))
    # fh_out.createDimension("lon", len(lons))
    #
    # outVar = fh_out.createVariable('lat', float, ('lat'))
    # outVar.setncatts({"units": "degree_north"})
    # outVar[:] = lats[:]
    # outVar = fh_out.createVariable('lon', float, ('lon'))
    # outVar.setncatts({"units": "degree_east"})
    # outVar[:] = lons[:]
    #
    # outVar = fh_out.createVariable("cdl", float, ("lat", "lon"))
    # outVar[:] = ma.masked_less(cdl_values, 0)
    #
    # fh_out.close()


if __name__ == "__main__":
    cdl_convert_to_nc("raw_data/cdl/2008_30m_cdls", "2008_30m_cdls.img",
                      "processed_data/cdl/30m/")
