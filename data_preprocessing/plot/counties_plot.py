#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from bs4 import BeautifulSoup
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import numpy as np
import seaborn as sns


# colors = sns.color_palette("RdYlBu", 10).as_hex()
colors = ['#cdeaf3', '#9bcce2', '#fff1aa', '#fece7f', '#fa9b58', '#ee613e', '#d22b27']


def counties_plot(data_dict, savepath, quantiles):
    """
    For the most part, reformatting of
    https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/6%20result_analysis/yield_map.py
    """
    # load the svg file
    svg = Path('../../processed_data/counties/counties.svg').open('r').read()
    # Load into Beautiful Soup
    soup = BeautifulSoup(svg, features="html.parser")
    # Find counties
    paths = soup.findAll('path')

    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1' \
                 ';stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start' \
                 ':none;stroke-linejoin:bevel;fill:'

    for p in paths:
        if p['id'] not in ["State_Lines", "separator"]:
            try:
                rate = data_dict[p['id']]
            except KeyError:
                continue
            if rate > quantiles[0.95]:
                color_class = 6
            elif rate > quantiles[0.8]:
                color_class = 5
            elif rate > quantiles[0.6]:
                color_class = 4
            elif rate > quantiles[0.4]:
                color_class = 3
            elif rate > quantiles[0.2]:
                color_class = 2
            elif rate > quantiles[0.05]:
                color_class = 1
            else:
                color_class = 0

            color = colors[color_class]
            p['style'] = path_style + color
    soup = soup.prettify()
    with savepath.open('w') as f:
        f.write(soup)


def save_colorbar(savedir, quantiles):
    """
    For the most part, reformatting of
    https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/6%20result_analysis/yield_map.py
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.02, 0.8])

    cmap = mpl.colors.ListedColormap(colors[1:-1])

    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])

    bounds = [quantiles[x] for x in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]]

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   # to use 'extend', you must
                                   # specify two extra boundaries:
                                   boundaries=[quantiles[0.0]] + bounds + [quantiles[1.0]],
                                   extend='both',
                                   ticks=bounds,  # optional
                                   spacing='proportional',
                                   orientation='vertical')
    plt.savefig('{}/colorbar.jpg'.format(savedir), dpi=300, bbox_inches='tight')


def process_yield_data():
    important_columns = ['Year', 'State ANSI', 'County ANSI', 'Value']
    yield_data = pd.read_csv('../../processed_data/crop_yield/yield_data.csv').dropna(
        subset=important_columns, how='any')[['Year', 'State ANSI', 'County ANSI', 'Value']]
    yield_data.columns = ['Year', 'State', 'County', 'Value']
    yield_per_year_dic = defaultdict(dict)

    for yd in yield_data.itertuples():
        year, state, county, value = yd.Year, yd.State, int(yd.County), yd.Value
        state = str(state).zfill(2)
        county = str(county).zfill(3)

        yield_per_year_dic[year][state+county] = value

    return yield_per_year_dic


if __name__ == '__main__':
    yield_data = process_yield_data()
    for year in range(2003, 2017):
        counties_plot(yield_data[year], Path('../../processed_data/crop_yield/plots/{}_yield.html'.format(year)))
        values = np.array(list(yield_data[year].values()))
        print(year, np.percentile(values, 0), np.percentile(values, 25), np.percentile(values, 50),
              np.percentile(values, 75), np.percentile(values, 100))
    save_colorbar('../../processed_data/crop_yield/plots')
