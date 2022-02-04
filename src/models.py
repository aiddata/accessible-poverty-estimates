"""
python 3.9

portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Run models based on OSM features and additional geospatial data


"""

import os
import sys
import configparser
import warnings
from joblib import dump, load

import pandas as pd

warnings.filterwarnings('ignore')


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project_dir = config["main"]["project_dir"]
osm_date = config["main"]["osm_date"]

data_dir = os.path.join(project_dir, 'data')

sys.path.insert(0, os.path.join(project_dir, 'src'))



import model_utils
import data_utils


import importlib
importlib.reload(model_utils)
importlib.reload(data_utils)


# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


show_plots = False

os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'results'), exist_ok=True)

# Scoring metrics
scoring = {
    'r2': data_utils.pearsonr2,
    'rmse': data_utils.rmse
}


# -------------------------------------


# Indicators of interest
# indicators = [
#     'Wealth Index',
#     'Education completed (years)',
#     'Access to electricity',
#     'Access to water (minutes)'
# ]
indicators = [
    'Wealth Index'
]


# load in dhs data
final_path =  os.path.join(data_dir, 'dhs_data.csv')
final_df = pd.read_csv(final_path)



# -------------------------------------
# OSM data prep


src_label = "dhs-buffers"

id_field = "DHSID"

# new osm data
osm_roads_file = os.path.join(data_dir, 'osm/features/{}_roads_{}.csv'.format(src_label, osm_date))
osm_buildings_file = os.path.join(data_dir, 'osm/features/{}_buildings_{}.csv'.format(src_label, osm_date))
osm_pois_file = os.path.join(data_dir, 'osm/features/{}_pois_{}.csv'.format(src_label, osm_date))
osm_traffic_file = os.path.join(data_dir, 'osm/features/{}_traffic_{}.csv'.format(src_label, osm_date))
osm_transport_file = os.path.join(data_dir, 'osm/features/{}_transport_{}.csv'.format(src_label, osm_date))


# Load OSM datasets
roads = pd.read_csv(osm_roads_file)
buildings = pd.read_csv(osm_buildings_file)
pois = pd.read_csv(osm_pois_file)
traffic = pd.read_csv(osm_traffic_file)
transport = pd.read_csv(osm_transport_file)

osm_df_list = [roads, buildings, pois, traffic, transport]

osm_df = pd.concat(osm_df_list, join="inner", axis=1)

osm_df = osm_df.loc[:, ~osm_df.columns.duplicated()]

osm_df = osm_df[[i for i in osm_df.columns if "_roads_nearest-osmid" not in i]]


print("Shape of osm dataframe: {}".format(osm_df.shape))

# Get list of columns
osm_cols = list(osm_df.columns[1:])

osm_cols = [i for i in osm_df if osm_df[i].min() != osm_df[i].max() and i != id_field]
osm_df = osm_df[[id_field] + osm_cols]

print("Shape of osm dataframe after drops: {}".format(osm_df.shape))
print('osm_cols:', osm_cols)


# only subset of OSM features related to basic all building/road footprints
footprint_cols = [i for i in osm_cols if i.startswith(('all_roads_', 'all_buildings'))]



# -------------------------------------
# geoquery spatial data prep

# join GeoQuery spatial data to osm_data
geoquery_path = os.path.join(data_dir, 'merge_phl_dhs_buffer.csv')
geoquery_df = pd.read_csv(geoquery_path)
geoquery_df.fillna(-999, inplace=True)

# print(geoquery_df.columns.to_list())

all_geoquery_cols = [i for i in geoquery_df.columns if len(i.split('.')) == 3]

for c1 in set([i[:i.index('categorical')] for i in geoquery_df.columns if 'categorical' in i]):
    for c2 in [i for i in geoquery_df.columns if i.startswith(c1) and not i.endswith('count')]:
        geoquery_df[c2] = geoquery_df[c2] / geoquery_df[c1 + 'categorical_count']

for y in range(2013, 2020):
    geoquery_df[f'esa_landcover.{y}.categorical_cropland'] = geoquery_df[[f'esa_landcover.{y}.categorical_irrigated_cropland', f'esa_landcover.{y}.categorical_rainfed_cropland', f'esa_landcover.{y}.categorical_mosaic_cropland']].sum(axis=1)

all_geoquery_cols = [i for i in geoquery_df.columns if len(i.split('.')) == 3]

# sub_geoquery_cols = ['udel_precip_v501_mean.2015.mean', 'udel_precip_v501_sum.2015.sum',  'udel_air_temp_v501_mean.2015.mean',  'srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'oco2.2015.mean', 'ltdr_avhrr_ndvi_v5_yearly.2015.mean', 'gpw_v4r11_density.2015.mean', 'gpw_v4r11_count.2015.sum',  'esa_landcover.2015.categorical_urban', 'esa_landcover.2015.categorical_water_bodies', 'esa_landcover.2015.categorical_forest', 'esa_landcover.2015.categorical_cropland', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'accessibility_to_cities_2015_v1.0.mean']
sub_geoquery_cols = ['srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'accessibility_to_cities_2015_v1.0.mean', 'gpw_v4r11_density.2015.mean', 'gpw_v4r11_count.2015.sum']

for y in range(2015,2018):
    sub_geoquery_cols.extend(
        [f'viirs.{y}.mean', f'viirs.{y}.min', f'viirs.{y}.max', f'viirs.{y}.sum',
        f'udel_precip_v501_mean.{y}.mean', f'udel_precip_v501_sum.{y}.sum',  f'udel_air_temp_v501_mean.{y}.mean',
        f'oco2.{y}.mean',
        f'ltdr_avhrr_ndvi_v5_yearly.{y}.mean',
        f'esa_landcover.{y}.categorical_urban', f'esa_landcover.{y}.categorical_water_bodies', f'esa_landcover.{y}.categorical_forest', f'esa_landcover.{y}.categorical_cropland']
    )



ntl_cols = ['viirs.2017.mean', 'viirs.2017.min', 'viirs.2017.max', 'viirs.2017.sum']


all_osm_cols = osm_cols
sub_osm_cols = [i for i in osm_cols if i.startswith(('all_buildings_', 'all_roads_')) and i != 'all_roads_count']


all_data_cols = all_osm_cols + all_geoquery_cols
sub_data_cols = sub_osm_cols + sub_geoquery_cols


spatial_df = osm_df.merge(geoquery_df, on=id_field, how="left")


all_data_df = spatial_df[[id_field] + all_data_cols]

all_data_df = all_data_df.merge(final_df[[id_field, 'Wealth Index']], on=id_field, how='left')

all_data_path = os.path.join(data_dir, 'all_data.csv')
all_data_df.to_csv(all_data_path)



all_data_df = all_data_df.loc[all_data_df['Wealth Index'].notnull()].copy()

for i in all_data_df.columns:
    na = all_data_df[i].isna().sum()
    if na > 0:
        print(i, all_data_df[i].isna().sum())



search_type = 'grid'

# -----------------------------------------------------------------------------
# Explore population distribution and relationships

data_utils.plot_hist(
    all_data_df['gpw_v4r11_count.2015.sum'],
    title='Distribution of Total Population',
    x_label='Total Population',
    y_label='Number of Clusters',
    output_file=os.path.join(data_dir, 'results', f'0_pop_hist.png'),
    show=show_plots
)

data_utils.plot_regplot(
    all_data_df,
    'Wealth Index',
    'Population',
    'gpw_v4r11_count.2015.sum',
    output_file=os.path.join(data_dir, 'results', f'1_pop_wealth_corr.png'),
    show=show_plots
)


# -----------------------------------------------------------------------------
# NTL only

data_utils.plot_regplot(
    data=all_data_df,
    x_label='Wealth Index',
    y_label='Average Nightlight Intensity',
    y_var='viirs.2017.mean',
    output_file=os.path.join(data_dir, 'results', f'2_ntl_wealth_regplot.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=ntl_cols,
    indicator='Wealth Index',
    method='pearsons',
    figsize=(8,6),
    output_file=os.path.join(data_dir, 'results', f'3_ntl_cols_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=ntl_cols,
    indicator='Wealth Index',
    method='spearman',
    figsize=(8,6),
    output_file=os.path.join(data_dir, 'results', f'4_ntl_cols_spearman_corr.png'),
    show=show_plots
)

ntl_cv, ntl_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=ntl_cols,
    indicator_cols=indicators,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    clust_str=id_field,
    output_file=os.path.join(data_dir, 'results', f'5_ntl_model_'),
    show=show_plots
)

model_utils.save_model(ntl_cv, all_data_df, ntl_cols, 'Wealth Index', os.path.join(data_dir, 'models/ntl_only_best.joblib'))


# -----------------------------------------------------------------------------
# all OSM only


data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_osm_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'6_osm_only_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_osm_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'7_osm_only_spearman_corr.png'),
    show=show_plots
)

osm_only_cv, osm_only_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=all_osm_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'8_osm_only_model_'),
    show=show_plots
)

model_utils.save_model(osm_only_cv, all_data_df, all_osm_cols, 'Wealth Index', os.path.join(data_dir, 'models/osm_only_best.joblib'))


# -----------------------------------------------------------------------------
# all OSM + NTL


all_osm_ntl_cols = all_osm_cols + ntl_cols

data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_osm_ntl_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'9_all_osm_cols_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_osm_ntl_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'10_all_osm_cols_spearman_corr.png'),
    show=show_plots
)

all_osm_cv, all_osm_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=all_osm_ntl_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'11_all_osm_ntl_model_'),
    show=show_plots
)

model_utils.save_model(all_osm_cv, all_data_df, all_osm_ntl_cols, 'Wealth Index', os.path.join(data_dir, 'models/all_osm_ntl_best.joblib'))


# -----------------------------------------------------------------------------
# sub OSM + NTL


sub_osm_ntl_cols = sub_osm_cols + ntl_cols

data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_osm_ntl_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'12_sub_osm_ntl_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_osm_ntl_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'13_sub_osm_ntl_spearman_corr.png'),
    show=show_plots
)

sub_osm_cv, sub_osm_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=sub_osm_ntl_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'14_sub_osm_ntl_model_'),
    show=show_plots
)

model_utils.save_model(sub_osm_cv, all_data_df, sub_osm_ntl_cols, 'Wealth Index', os.path.join(data_dir, 'models/sub_osm_best.joblib'))


# -----------------------------------------------------------------------------
# NTL + sub geoquery


data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_geoquery_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'15_sub_geoquery_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_geoquery_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'16_sub_geoquery_spearman_corr.png'),
    show=show_plots
)

subgeo_cv, subgeo_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=sub_geoquery_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'17_sub_geoquery_model_'),
    show=show_plots
)

model_utils.save_model(subgeo_cv, all_data_df, sub_geoquery_cols, 'Wealth Index', os.path.join(data_dir, 'models/sub_geoquery_best.joblib'))


# -----------------------------------------------------------------------------
# NTL + sub OSM + sub geoquery


data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_data_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'18_sub_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=sub_data_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'19_sub_spearman_corr.png'),
    show=show_plots
)


sub_cv, sub_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=sub_data_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'20_sub_model_'),
    show=show_plots
)


model_utils.save_model(sub_cv, all_data_df, sub_data_cols, 'Wealth Index', os.path.join(data_dir, 'models/sub_best.joblib'))




# -----------------------------------------------------------------------------


print(f"NTL best estimator: {ntl_cv.best_estimator_}")
print(f"OSM only best estimator: {osm_only_cv.best_estimator_}")
print(f"All OSM best estimator: {all_osm_cv.best_estimator_}")
print(f"Sub OSM best estimator: {sub_osm_cv.best_estimator_}")
print(f"Sub best estimator: {sub_cv.best_estimator_}")
print(f"Sub GeoQuery best estimator: {subgeo_cv.best_estimator_}")


# -----------------------------------------------------------------------------
# feature reduction

#return a dictionary that identifes a a list of variables for which each variables has a correlation above the specified threshold
#will only return variables that had at least one other variable with the specified correlation threshold
correlated_ivs, corr_matrix = data_utils.corr_finder(all_data_df, .85)

#subset data based on correlation but make sure that specifically desired covariates are still within the group.
#ensure that even if road/building data can have a low correlation with each other, only one from each group is used per model evaluation
remove_corrs = correlated_ivs['viirs.2016.mean']  + correlated_ivs['gpw_v4r11_count.2015.sum']  + correlated_ivs['all_roads_length'] + correlated_ivs['all_buildings_ratio']
to_keep = ['viirs.2016.mean','all_buildings_totalarea', 'all_buildings_count', 'all_roads_nearestdist', 'gpw_v4r11_count.2015.sum']
to_remove_dict = {
    'correlation': [label for label in remove_corrs if label not in to_keep],
    'manual': [],
    # 'osm': [i for i in osm_cols if not i.startswith(('all_buildings_', 'all_roads_'))]
}
to_remove_list = [j for i in to_remove_dict.values() for j in i]


new_features = [i for i in sub_data_cols if i not in to_remove_list]


#create a dataframe based on feature importance the df is ordered from most important to least important feature.
reduction_df = model_utils.rf_feature_importance_dataframe(sub_cv, all_data_df[new_features], all_data_df[indicators])

## subset data based on desired feature importance feature importance
thresh = .01
remove_importance = []
for row, ser in reduction_df.iterrows():
    for idx, val in ser.iteritems():
        if (val < thresh): #if the variable correlates past/at the threshold
            remove_importance.append(row)


important_features = [i for i in new_features if i not in remove_importance]
important_features = ['all_roads_length', 'all_roads_nearestdist', 'all_buildings_ratio', 'distance_to_coast_236.na.mean', 'accessibility_to_cities_2015_v1.0.mean', 'gpw_v4r11_density.2015.mean', 'viirs.2017.min', 'udel_precip_v501_sum.2017.sum', 'udel_air_temp_v501_mean.2017.mean', 'esa_landcover.2017.categorical_water_bodies', 'esa_landcover.2017.categorical_cropland',  'oco2.2017.mean', 'ltdr_avhrr_ndvi_v5_yearly.2017.mean',  'viirs.2017.mean',  'esa_landcover.2017.categorical_urban', ]

final_cv, final_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=important_features,
    indicator_cols=indicators,
    search_type="grid",
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(data_dir, 'results', f'21_final_model_'),
    show=show_plots
)


model_utils.save_model(final_cv, all_data_df, important_features, 'Wealth Index', os.path.join(data_dir, 'models/final_best.joblib'))
