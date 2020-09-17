# Context-aware Deep Representation Learning for Geo-spatiotemporal Analysis
Code for ICDM 2020 paper Context-aware Deep Representation Learning for Geo-spatiotemporal Analysis.

## Data Preprocessing

### Data Sources
1. County-level soybean yields (year 2003 to 2018) is downloaded from [USDA NASS Quick Stats Database](https://quickstats.nass.usda.gov/).
2. Landcover class is from the [MODIS product MCD12Q1](https://lpdaac.usgs.gov/products/mcd12q1v006/) and downloaded from Google Earth Engine. **gee_county_lc.py** and **gee_landcover.py** are the files to call Google Earth Engine and download the data. County boundaries from [Google's fusion table](https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1) are utilized to download the landcover class data for each county separately.

Input:

1. Vegetation indices including NDVI and EVI are from [MODIS product MOD13A3](https://lpdaac.usgs.gov/products/mod13a3v006/) and downloaded from [AρρEEARS](https://lpdaacsvc.cr.usgs.gov/appeears/).
2. Precipitation is from the [PRISM dataset](http://www.prism.oregonstate.edu/).
3. Land surface temperature is from [MODIS product MOD11A1]() and downloaded from [AρρEEARS](https://lpdaacsvc.cr.usgs.gov/appeears/).
4. Elevation is from the NASA Shuttle Radar Topography Mission Global 30 m product and downloaded from [AρρEEARS](https://lpdaacsvc.cr.usgs.gov/appeears/).
5. Soil properties including soil sand, silt, and clay fractions are from the [STATSGO](https://catalog.data.gov/dataset/statsgo-soil-polygons) data base.

### Preprocessing
Data from various sources are first converted to a unified format [netCDF4](https://unidata.github.io/netcdf4-python/netCDF4/index.html) with their original resolutions being kept. They are then rescaled to the MODIS product grid at 1 km resolution.

## Experiment Data Generation
Quadruplet sampling code is contained in folder **data_preprocessing/sample_quadruplets**. Functions are then called by **generate_experiment_data.py** to generate experiment data. 

## Modeling
We provide code here for the context-aware representation learning model and all baselines mentioned in the paper, including traditional models for scalar inputs, deep gausian models, cnn-lstm and c3d.

A few examples of commands to train the models:
1. attention model - semisupervised:
```console
python ./crop_yield_train_semi_transformer.py --neighborhood-radius 25 --distant-radius 100 --weight-decay 0.0 --tilenet-margin 50 --tilenet-l2 0.2 --tilenet-ltn 0.001 --tilenet-zdim 256 --attention-layer 1 --attention-dff 512 --sentence-embedding simple_average --dropout 0.2 --unsup-weight 0.2 --patience 9999 --feature all --feature-len 9 --year 2018 --ntsteps 7 --train-years 10 --query-type combine
```
2. attention model - supervised:
```console
python ./crop_yield_train_semi_transformer.py --neighborhood-radius 25 --distant-radius 100 --weight-decay 0.0 --tilenet-margin 50 --tilenet-l2 0.2 --tilenet-ltn 0.001 --tilenet-zdim 256 --attention-layer 1 --attention-dff 512 --sentence-embedding simple_average --dropout 0.2 --unsup-weight 0.0 --patience 9999 --feature all --feature-len 9 --year 2018 --ntsteps 7 --train-years 10 --query-type combine
```
&nbsp;&nbsp;When query type is set as "combine", the hybrid attention mechanism introduced in the ICDM 2020 papaer is adopted. You can test other query types ("global", "fixed", "separate") on your data as well.

3. c3d
```console
python ./crop_yield_train_c3d.py --patience 9999 --feature all --feature-len 9 --year 2018 --ntsteps 7 --train-years 10
```
4. cnn-lstm
```console
python ./crop_yield_train_cnn_lstm.py --patience 9999 --feature all --feature-len 9 --year 2018 --ntsteps 7 --train-years 10 --tilenet-zdim 256 --lstm-inner 512
```
5. deep gaussian
```console
python ./crop_yield_deep_gaussian.py --type cnn --time 7 --train-years 10
```
6. traditional models
```console
python ./crop_yield_no_spatial.py --predict no_spatial --train-years 10
```

## Cite this work

## License
MIT licensed. See the LICENSE file for details.
