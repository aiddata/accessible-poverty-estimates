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


# all_geoquery_cols = ['wb_aid.na.sum', 'wdpa_iucn_cat_201704.na.categorical_count', 'wdpa_iucn_cat_201704.na.categorical_unprotected', 'wdpa_iucn_cat_201704.na.categorical_ia', 'wdpa_iucn_cat_201704.na.categorical_ib', 'wdpa_iucn_cat_201704.na.categorical_ii', 'wdpa_iucn_cat_201704.na.categorical_iii', 'wdpa_iucn_cat_201704.na.categorical_iv', 'wdpa_iucn_cat_201704.na.categorical_v', 'wdpa_iucn_cat_201704.na.categorical_vi', 'wdpa_iucn_cat_201704.na.categorical_not_applicable', 'wdpa_iucn_cat_201704.na.categorical_not_assigned', 'wdpa_iucn_cat_201704.na.categorical_not_reported', 'wdpa_iucn_cat_201704.na.categorical_mix', 'wdpa_iucn_cat_201704.na.count', 'viirs.2013.mean', 'viirs.2014.mean', 'viirs.2015.mean', 'viirs.2016.mean', 'viirs.2017.mean', 'viirs.2018.mean', 'viirs.2019.mean', 'viirs.2020.mean', 'viirs.2013.min', 'viirs.2014.min', 'viirs.2015.min', 'viirs.2016.min', 'viirs.2017.min', 'viirs.2018.min', 'viirs.2019.min', 'viirs.2020.min', 'viirs.2013.max', 'viirs.2014.max', 'viirs.2015.max', 'viirs.2016.max', 'viirs.2017.max', 'viirs.2018.max', 'viirs.2019.max', 'viirs.2020.max', 'viirs.2013.sum', 'viirs.2014.sum', 'viirs.2015.sum', 'viirs.2016.sum', 'viirs.2017.sum', 'viirs.2018.sum', 'viirs.2019.sum', 'viirs.2020.sum', 'udel_precip_v501_mean.2013.mean', 'udel_precip_v501_mean.2014.mean', 'udel_precip_v501_mean.2015.mean', 'udel_precip_v501_mean.2016.mean', 'udel_precip_v501_mean.2017.mean', 'udel_precip_v501_sum.2013.sum', 'udel_precip_v501_sum.2014.sum', 'udel_precip_v501_sum.2015.sum', 'udel_precip_v501_sum.2016.sum', 'udel_precip_v501_sum.2017.sum', 'udel_air_temp_v501_mean.2013.mean', 'udel_air_temp_v501_mean.2014.mean', 'udel_air_temp_v501_mean.2015.mean', 'udel_air_temp_v501_mean.2016.mean', 'udel_air_temp_v501_mean.2017.mean', 'srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'oco2.2015.mean', 'oco2.2016.mean', 'oco2.2017.mean', 'oco2.2018.mean', 'oco2.2019.mean', 'oco2.2020.mean', 'ltdr_avhrr_ndvi_v5_yearly.2013.mean', 'ltdr_avhrr_ndvi_v5_yearly.2014.mean', 'ltdr_avhrr_ndvi_v5_yearly.2015.mean', 'ltdr_avhrr_ndvi_v5_yearly.2016.mean', 'ltdr_avhrr_ndvi_v5_yearly.2017.mean', 'ltdr_avhrr_ndvi_v5_yearly.2018.mean', 'ltdr_avhrr_ndvi_v5_yearly.2019.mean', 'ltdr_avhrr_ndvi_v5_yearly.2020.mean', 'gpw_v4r11_density.2015.mean', 'gpw_v4r11_density.2020.mean', 'gpw_v4r11_count.2015.sum', 'gpw_v4r11_count.2020.sum', 'esa_landcover_v207.2013.categorical_count', 'esa_landcover_v207.2013.categorical_mosaic_cropland', 'esa_landcover_v207.2013.categorical_rainfed_cropland', 'esa_landcover_v207.2013.categorical_urban', 'esa_landcover_v207.2013.categorical_water_bodies', 'esa_landcover_v207.2013.categorical_forest', 'esa_landcover_v207.2013.categorical_irrigated_cropland', 'esa_landcover_v207.2013.categorical_no_data', 'esa_landcover_v207.2013.categorical_bare_areas', 'esa_landcover_v207.2013.categorical_sparse_vegetation', 'esa_landcover_v207.2013.categorical_grassland', 'esa_landcover_v207.2013.categorical_wetland', 'esa_landcover_v207.2013.categorical_shrubland', 'esa_landcover_v207.2013.categorical_snow_ice', 'esa_landcover_v207.2014.categorical_count', 'esa_landcover_v207.2014.categorical_mosaic_cropland', 'esa_landcover_v207.2014.categorical_rainfed_cropland', 'esa_landcover_v207.2014.categorical_urban', 'esa_landcover_v207.2014.categorical_water_bodies', 'esa_landcover_v207.2014.categorical_forest', 'esa_landcover_v207.2014.categorical_irrigated_cropland', 'esa_landcover_v207.2014.categorical_no_data', 'esa_landcover_v207.2014.categorical_bare_areas', 'esa_landcover_v207.2014.categorical_sparse_vegetation', 'esa_landcover_v207.2014.categorical_grassland', 'esa_landcover_v207.2014.categorical_wetland', 'esa_landcover_v207.2014.categorical_shrubland', 'esa_landcover_v207.2014.categorical_snow_ice', 'esa_landcover_v207.2015.categorical_count', 'esa_landcover_v207.2015.categorical_mosaic_cropland', 'esa_landcover_v207.2015.categorical_rainfed_cropland', 'esa_landcover_v207.2015.categorical_urban', 'esa_landcover_v207.2015.categorical_water_bodies', 'esa_landcover_v207.2015.categorical_forest', 'esa_landcover_v207.2015.categorical_irrigated_cropland', 'esa_landcover_v207.2015.categorical_no_data', 'esa_landcover_v207.2015.categorical_bare_areas', 'esa_landcover_v207.2015.categorical_sparse_vegetation', 'esa_landcover_v207.2015.categorical_grassland', 'esa_landcover_v207.2015.categorical_wetland', 'esa_landcover_v207.2015.categorical_shrubland', 'esa_landcover_v207.2015.categorical_snow_ice', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'accessibility_to_cities_2015_v1.0.mean']

ntl_cols = ['viirs.2017.mean', 'viirs.2017.min', 'viirs.2017.max', 'viirs.2017.sum']

geoquery_cols = ['wb_aid.na.sum',  'udel_precip_v501_mean.2017.mean', 'udel_precip_v501_sum.2017.sum',  'udel_air_temp_v501_mean.2017.mean',  'srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'oco2.2017.mean', 'ltdr_avhrr_ndvi_v5_yearly.2017.mean', 'gpw_v4r11_density.2015.mean', 'gpw_v4r11_count.2015.sum',  'esa_landcover_v207.2013.categorical_count', 'esa_landcover_v207.2013.categorical_rainfed_cropland', 'esa_landcover_v207.2013.categorical_urban', 'esa_landcover_v207.2013.categorical_water_bodies', 'esa_landcover_v207.2013.categorical_forest', 'esa_landcover_v207.2013.categorical_irrigated_cropland', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'accessibility_to_cities_2015_v1.0.mean']

all_data_cols = osm_cols + ntl_cols + geoquery_cols

print(len(all_data_cols))



spatial_df = osm_df.merge(geoquery_df, on=id_field, how="left")


all_data_df = spatial_df[[id_field] + all_data_cols]

all_data_df = all_data_df.merge(final_df[[id_field, 'Wealth Index']], on=id_field, how='left')

all_data_path = os.path.join(data_dir, 'all_data.csv')
all_data_df.to_csv(all_data_path)



all_data_df = all_data_df.loc[all_data_df['Wealth Index'].notnull()].copy()

for i in all_data_df.columns:
    na = all_data_df[i].isna().sum()
    if na > 0: print(i, all_data_df[i].isna().sum())



# -----------------------------------------------------------------------------
# Explore population distribution and relationships

data_utils.plot_hist(
    all_data_df['gpw_v4r11_count.2015.sum'],
    title='Distribution of Total Population',
    x_label='Total Population',
    y_label='Number of Clusters',
    output_file=os.path.join(data_dir, 'results', f'pop_hist.png'),
    show=show_plots
)

data_utils.plot_regplot(
    all_data_df,
    'Wealth Index',
    'Population',
    'gpw_v4r11_count.2015.sum',
    output_file=os.path.join(data_dir, 'results', f'pop_wealth_corr.png'),
    show=show_plots
)


# -----------------------------------------------------------------------------
# NTL only models

data_utils.plot_regplot(
    data=all_data_df,
    x_label='Wealth Index',
    y_label='Average Nightlight Intensity',
    y_var='viirs.2017.mean',
    output_file=os.path.join(data_dir, 'results', f'ntl_wealth_regplot.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=ntl_cols,
    indicator='Wealth Index',
    method='pearsons',
    figsize=(8,6),
    output_file=os.path.join(data_dir, 'results', f'ntl_cols_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=ntl_cols,
    indicator='Wealth Index',
    method='spearman',
    figsize=(8,6),
    output_file=os.path.join(data_dir, 'results', f'ntl_cols_spearman_corr.png'),
    show=show_plots
)

ntl_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=ntl_cols,
    indicator_cols=indicators,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type='random',
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    clust_str=id_field,
    output_file=os.path.join(data_dir, 'results', f'ntl_model_'),
    show=show_plots
)



# -----------------------------------------------------------------------------
# OSM + NTL models


osm_ntl_cols = osm_cols + ntl_cols

data_utils.plot_corr(
    data=all_data_df,
    features_cols=osm_ntl_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'osm_cols_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=osm_ntl_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'osm_cols_spearman_corr.png'),
    show=show_plots
)

osm_cv, osm_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=osm_ntl_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type='random',
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2
)


# define X,y for all data
osm_X = all_data_df[osm_ntl_cols]
osm_y = all_data_df['Wealth Index'].tolist()


# refit cv model with all data
osm_best = osm_cv.best_estimator_.fit(osm_X, osm_y)

# save model
osm_model_path = os.path.join(data_dir, 'models/osm_ntl_best.joblib') #added ntl to list
dump(osm_best, osm_model_path)




# -----------------------------------------------------------------------------
# NTL + OSM + spatial models


data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_data_cols,
    indicator='Wealth Index',
    method='pearsons',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'allcols_pearsons_corr.png'),
    show=show_plots
)

data_utils.plot_corr(
    data=all_data_df,
    features_cols=all_data_cols,
    indicator='Wealth Index',
    method='spearman',
    max_n=50,
    figsize=(10,13),
    output_file=os.path.join(data_dir, 'results', f'allcols_spearman_corr.png'),
    show=show_plots
)


all_cv, all_predictions = model_utils.evaluate_model(
    data=all_data_df,
    feature_cols=all_data_cols,
    indicator_cols=indicators,
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type='random',
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2
)



# -----------------------------------------------------------------------------
# feature reduction

#return a dictionary that identifes a a list of variables for which each variables has a correlation of .7 or higher
#will only return variables that had at least one other variable with an .85 correlation
correlated_ivs, corr_matrix = data_utils.corr_finder(all_data_df, .8)

#subset data based on correlation but make sure that specifically desired covariates are still within the group.
#ensure that even if road/building data can have a low correlation with each other, only one from each group is used per model evaluation
remove_corrs = correlated_ivs['viirs.2017.mean']  + correlated_ivs['gpw_v4r11_count.2015.sum']  + correlated_ivs['all_roads_length'] + correlated_ivs['all_buildings_ratio']
to_keep = ['viirs.2017.mean','all_buildings_totalarea', 'all_buildings_count', 'all_roads_count', 'all_roads_nearestdist', 'gpw_v4r11_count.2015.sum']
remove_corrs = [label for label in remove_corrs if label not in to_keep]
remove_corrs = remove_corrs + [i for i in osm_cols if not i.startswith(('all_buildings_', 'all_roads_'))]


updated_df, new_features = data_utils.subset_dataframe(all_data_df, all_data_cols, remove_corrs)
print('Features tested on first round:', new_features)
print('Number of features tested on first round', len(new_features))


test1_cv, test1_predictions = model_utils.evaluate_model(
    data=updated_df,
    feature_cols=new_features,
    indicator_cols=indicators,
    search_type="grid",
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    n_splits=4,
    n_iter=10,
    plot_importance=True,
    verbose=2
)


#create a dataframe based on feature importance the df is ordered from most important to least important feature.
df = model_utils.rf_feature_importance_dataframe(test1_cv, updated_df[new_features], updated_df[indicators])


## subset data based on desired feature importance feature importance
thresh = .01
remove_importance = []
for row, ser in df.iterrows():
    for idx, val in ser.iteritems():
        if (val < thresh): #if the variable correlates past/at the threshold
            remove_importance.append(row)

print('The removed columns from feature importance subsetting are:', remove_importance)

updated_df, new_features = data_utils.subset_dataframe(updated_df, new_features, remove_importance)
print('Features used by subsetting feature importance <', thresh,':', new_features)

test1_cv, test1_predictions = model_utils.evaluate_model(
    data=updated_df,
    feature_cols=new_features,
    indicator_cols=indicators,
    search_type="grid",
    clust_str=id_field,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    n_splits=4,
    n_iter=10,
    plot_importance=True,
    verbose=2
)



#create a dataframe based on feature importance after the second model has run
df = model_utils.rf_feature_importance_dataframe(test1_cv, updated_df[new_features], updated_df[indicators])
print('Feature importance of final model:', df)


# #define X,y for all data
test1_X = updated_df[new_features]
test1_y = updated_df['Wealth Index'].tolist()


 # refit cv model with all data
test1_best = test1_cv.best_estimator_.fit(test1_X, test1_y)

# save model
test1_model_path = os.path.join(data_dir, 'models/test1_best.joblib')
dump(test1_best, test1_model_path)

