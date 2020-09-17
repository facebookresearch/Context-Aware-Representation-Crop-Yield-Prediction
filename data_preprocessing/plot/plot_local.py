#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np


# mean of lats: 40.614586, mean of lons: -121.24792
def plot_local(in_file, x_axis, y_axis):
    fh = Dataset(in_file, 'r')
    lats = fh.variables['lat'][:]
    lons = fh.variables['lon'][:]
    x_indices = [(np.abs(lons-i)).argmin() for i in x_axis]
    y_indices = [(np.abs(lats-i)).argmin() for i in y_axis]
    for v in fh.variables.keys():
        if v not in ['lat', 'lon']:
            values = fh.variables[v][:]
            plt.imshow(values, interpolation='none', cmap=plt.get_cmap("jet"))
            plt.title(v)
            plt.gca().set_xticks(x_indices)
            plt.gca().set_yticks(y_indices)
            plt.gca().set_xticklabels(x_axis)
            plt.gca().set_yticklabels(y_axis)
            plt.colorbar()
            plt.savefig('../../processed_data/local/ca_20190604/{}.jpg'.format(v))
            plt.close()


def plot_landsat(in_file, x_axis, y_axis):
    fh = Dataset(in_file, 'r')
    lats = fh.variables['lat'][:][::-1]
    lons = fh.variables['lon'][:]
    x_indices = [(np.abs(lons - i)).argmin() for i in x_axis]
    y_indices = [(np.abs(lats - i)).argmin() for i in y_axis]
    titles = ["Band 1 Ultra Blue", "Band 2 Blue", "Band 3 Green",
              "Band 4 Red", "Band 5 Near Infrared",
              "Band 6 Shortwave Infrared 1", "Band 7 Shortwave Infrared 2"]
    for title, v in zip(titles, range(1, 8)):
        values = np.flipud(fh.variables['band{}'.format(v)][:])
        plt.imshow(values, interpolation='none', cmap=plt.get_cmap("jet"), vmin=0, vmax=10000)
        plt.title(title)
        plt.gca().set_xticks(x_indices)
        plt.gca().set_yticks(y_indices)
        plt.gca().set_xticklabels(x_axis)
        plt.gca().set_yticklabels(y_axis)
        plt.colorbar()
        plt.savefig('../../processed_data/local/ca_20190604/band{}.jpg'.format(v))
        plt.close()


if __name__ == '__main__':
    y_axis = [41.20, 40.95, 40.70, 40.45, 40.20]
    x_axis = [-122.0, -121.75, -121.5, -121.25, -121.0, -120.75, -120.5]
    plot_local('../../processed_data/local/ca_20190604/elevation.nc', x_axis, y_axis)
    plot_local('../../processed_data/local/ca_20190604/lai.nc', x_axis, y_axis)
    plot_local('../../processed_data/local/ca_20190604/lst.nc', x_axis, y_axis)
    plot_local('../../processed_data/local/ca_20190604/nws_precip.nc', x_axis, y_axis)
    plot_local('../../processed_data/local/ca_20190604/soil_fraction.nc', x_axis, y_axis)
    plot_local('../../processed_data/local/ca_20190604/soil_moisture.nc', x_axis, y_axis)
    plot_landsat('../../processed_data/local/ca_20190604/landsat.nc', x_axis, y_axis)

