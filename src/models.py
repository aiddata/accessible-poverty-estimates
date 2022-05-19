"""
python 3.9

portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Run models based on OSM features and additional geospatial data


"""

import os
import sys
import configparser
import warnings
import json
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

show_plots = False


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

data_dir = os.path.join(project_dir, 'data')

output_name = config[project]['output_name']

models_dir = os.path.join(data_dir, 'outputs', output_name, 'models')
results_dir = os.path.join(data_dir, 'outputs', output_name, 'results')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)





sys.path.insert(0, os.path.join(project_dir, 'src'))

import model_utils
import data_utils


# import importlib
# importlib.reload(model_utils)
# importlib.reload(data_utils)


# indicators = [
#     'Wealth Index',
#     'Education completed (years)',
#     'Access to electricity',
#     'Access to water (minutes)'
# ]
indicators = ['Wealth Index']


# Scoring metrics
scoring = {
    'r2': data_utils.pearsonr2,
    'rmse': data_utils.rmse
}

search_type = 'grid'

# number of folds for cross-validation
n_splits = 5







if 'combination' in config[project] and config[project]['combination'] == 'True':
    dhs_list = config[project]['project_list'].replace(' ', '').split(',')

else:
    dhs_list = [project]



project_data_dict = {}

for dhs_item in dhs_list:

    dhs_round = config[dhs_item]['dhs_round']
    country_utm_epsg_code = config[dhs_item]['country_utm_epsg_code']

    osm_date = config[dhs_item]["osm_date"]
    geom_id = config[dhs_item]["geom_id"]
    geom_label = config[dhs_item]["geom_label"]

    geoquery_data_file_name = config[dhs_item]["geoquery_data_file_name"]

    ntl_year = config[dhs_item]["ntl_year"]

    geospatial_variable_years = json.loads(config[dhs_item]['geospatial_variable_years'])



    # -------------------------------------
    # load in dhs data

    dhs_path =  os.path.join(data_dir, 'outputs', dhs_round, 'dhs_data.csv')
    raw_dhs_df = pd.read_csv(dhs_path)
    dhs_cols = [geom_id, 'latitude', 'longitude'] + indicators
    dhs_df = raw_dhs_df[dhs_cols]


    # -------------------------------------
    # OSM data prep

    osm_features_dir = os.path.join(data_dir, 'outputs', dhs_round, 'osm_features')

    # new osm data
    osm_roads_file = os.path.join(osm_features_dir, '{}_roads_{}.csv'.format(geom_label, osm_date))
    osm_buildings_file = os.path.join(osm_features_dir, '{}_buildings_{}.csv'.format(geom_label, osm_date))
    osm_pois_file = os.path.join(osm_features_dir, '{}_pois_{}.csv'.format(geom_label, osm_date))
    osm_traffic_file = os.path.join(osm_features_dir, '{}_traffic_{}.csv'.format(geom_label, osm_date))
    osm_transport_file = os.path.join(osm_features_dir, '{}_transport_{}.csv'.format(geom_label, osm_date))

    # Load OSM datasets
    roads = pd.read_csv(osm_roads_file)
    buildings = pd.read_csv(osm_buildings_file)
    pois = pd.read_csv(osm_pois_file)
    traffic = pd.read_csv(osm_traffic_file)
    transport = pd.read_csv(osm_transport_file)

    osm_df_list = [roads, buildings, pois, traffic, transport]

    raw_osm_df = pd.concat(osm_df_list, join="inner", axis=1)

    raw_osm_df = raw_osm_df.loc[:, ~raw_osm_df.columns.duplicated()]

    raw_osm_df = raw_osm_df[[i for i in raw_osm_df.columns if "_roads_nearest-osmid" not in i]]


    print("Shape of raw OSM dataframe: {}".format(raw_osm_df.shape))

    osm_drop_cols = [i for i in raw_osm_df if raw_osm_df[i].min() == raw_osm_df[i].max()]

    print("Dropping OSM columns (no variance): ", osm_drop_cols)

    osm_df = raw_osm_df.drop(columns=osm_drop_cols)

    print("Shape of OSM dataframe after drops: {}".format(osm_df.shape))


    # -------------------------------------
    # geoquery spatial data prep

    # join GeoQuery spatial data to osm_data
    geoquery_path = os.path.join(data_dir, 'outputs', dhs_round, f'{geoquery_data_file_name}.csv')
    geoquery_df = pd.read_csv(geoquery_path)
    geoquery_df.fillna(-999, inplace=True)


    for c1 in set([i[:i.index('categorical')] for i in geoquery_df.columns if 'categorical' in i]):
        for c2 in [i for i in geoquery_df.columns if i.startswith(c1) and not i.endswith('count')]:
            if c1 + 'categorical_count' in geoquery_df.columns:
                geoquery_df[c2] = geoquery_df[c2] / geoquery_df[c1 + 'categorical_count']


    esa_landcover_years = set([i.split('.')[1] for i in geoquery_df.columns.to_list() if 'esa_landcover' in i])
    for y in esa_landcover_years:
        geoquery_df[f'esa_landcover.{y}.categorical_cropland'] = geoquery_df[[f'esa_landcover.{y}.categorical_irrigated_cropland', f'esa_landcover.{y}.categorical_rainfed_cropland', f'esa_landcover.{y}.categorical_mosaic_cropland']].sum(axis=1)


    spatial_df = osm_df.merge(geoquery_df, on=geom_id, how="left")

    all_data_df = spatial_df.merge(dhs_df, on=geom_id, how='left')


    all_data_df = all_data_df.loc[all_data_df['Wealth Index'].notnull()].copy()

    for i in all_data_df.columns:
        na = all_data_df[i].isna().sum()
        if na > 0:
            print(i, all_data_df[i].isna().sum())



    # -------------------------------------
    # prepare feature lists


    # geoquery
    all_raw_geoquery_cols = [i for i in geoquery_df.columns if len(i.split('.')) == 3]

    all_geoquery_cols = [i.replace('.', '_') for i in all_raw_geoquery_cols]

    all_data_df.rename(dict(zip(all_raw_geoquery_cols, all_geoquery_cols)), axis=1, inplace=True)


    data_keep_geoquery_cols = ['accessibility_to_cities_2015_v1_0_mean', 'gpw_v4r11_density_2015_mean']

    data_drop_geoquery_cols = ['globalwindatlas_windspeed_na_mean', 'globalsolaratlas_pvout_na_mean', 'wb_aid_na_sum', 'wdpa_iucn_cat_201704_na_', 'distance_to_coast_236_na_mean', 'onshore_petroleum_v12_na_mean', 'gemdata_201708_na_sum', 'gdp_grid_na_sum', 'drugdata_categorical_201708_na_categorical_', 'distance_to_gold_v12_na_mean', 'distance_to_gemdata_201708_na_mean', 'distance_to_drugdata_201708_na_mean', 'dist_to_onshore_petroleum_v12_na_mean', 'diamond_distance_201708_na_mean', 'diamond_binary_201708_na_mean', 'ambient_air_pollution', 'gpm_precipitation', 'oco2']

    drop_years = [str(i) for i in range(2000,2022) if i not in geospatial_variable_years]

    year_drop_geoquery_cols = [i for i in all_geoquery_cols if any([j in i for j in drop_years])]

    temporal_main_geoquery_cols = [i for i in all_geoquery_cols if i in data_keep_geoquery_cols or not i.startswith(tuple(data_drop_geoquery_cols+year_drop_geoquery_cols))]


    atemporal_replacement_dict = dict([(i, i.replace(f'_{ntl_year}_', '_')) for i in temporal_main_geoquery_cols if not i in data_keep_geoquery_cols])

    main_geoquery_cols = list(atemporal_replacement_dict.values()) + data_keep_geoquery_cols
    all_data_df.rename(atemporal_replacement_dict, axis=1, inplace=True)



    # sub_geoquery_cols = ['srtm_slope_500m_na_mean', 'srtm_elevation_500m_na_mean', 'distance_to_coast_236_na_mean', 'dist_to_water_na_mean', 'accessibility_to_cities_2015_v1_0_mean', 'gpw_v4r11_density_2015_mean', 'gpw_v4r11_count_2015_sum',  'globalwindatlas_windspeed_na_mean', 'distance_to_gemdata_201708_na_mean', 'dist_to_onshore_petroleum_v12_na_mean']

    # for y in range(2015,2018):
    #     sub_geoquery_cols.extend(
    #         [f'viirs_{y}_mean', f'viirs_{y}_min', f'viirs_{y}_max', f'viirs_{y}_sum', f'viirs_{y}_median',
    #         f'udel_precip_v501_mean_{y}_mean', f'udel_precip_v501_sum_{y}_sum',  f'udel_air_temp_v501_mean_{y}_mean',
    #         f'oco2_{y}_mean',
    #         f'ltdr_avhrr_ndvi_v5_yearly_{y}_mean',
    #         f'esa_landcover_{y}_categorical_urban', f'esa_landcover_{y}_categorical_water_bodies', f'esa_landcover_{y}_categorical_forest', f'esa_landcover_{y}_categorical_cropland']
    #     )

    # sub_geoquery_cols = ['srtm_slope_500m_na_mean', 'srtm_elevation_500m_na_mean', 'dist_to_water_na_mean', 'accessibility_to_cities_2015_v1_0_mean', 'gpw_v4r11_density_2015_mean']

    # for y in geospatial_variable_years:
    #     sub_geoquery_cols.extend(
    #         [f'viirs_{y}_mean', f'viirs_{y}_min', f'viirs_{y}_max', f'viirs_{y}_sum', f'viirs_{y}_median',
    #         f'udel_precip_v501_mean_{y}_mean', f'udel_precip_v501_sum_{y}_sum',  f'udel_air_temp_v501_mean_{y}_mean',
    #         f'worldpop_pop_count_1km_mosaic_{y}_mean',
    #         f'oco2_{y}_mean',
    #         f'ltdr_avhrr_ndvi_v5_yearly_{y}_mean',
    #         f'esa_landcover_{y}_categorical_urban', f'esa_landcover_{y}_categorical_water_bodies', f'esa_landcover_{y}_categorical_forest', f'esa_landcover_{y}_categorical_cropland']
    #     )

    # sub_geoquery_cols = [i for i in sub_geoquery_cols if i in all_data_df]

    # sub_geoquery_cols = [f'viirs_{ntl_year}_max', f'viirs_{ntl_year}_median', 'srtm_elevation_500m_na_mean', f'udel_precip_v501_sum_{ntl_year}_sum', f'ltdr_avhrr_ndvi_v5_yearly_{ntl_year}_mean', 'dist_to_water_na_mean', 'accessibility_to_cities_2015_v1_0_mean', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean', f'esa_landcover_{ntl_year}_categorical_cropland', 'longitude', 'latitude', f'esa_landcover_{ntl_year}_categorical_urban', f'esa_landcover_{ntl_year}_categorical_forest', f'modis_lst_mod11c3_006_day_annual_mean_{ntl_year}_mean']

    # ntl_cols = [f'viirs_{ntl_year}_mean', f'viirs_{ntl_year}_min', f'viirs_{ntl_year}_max', f'viirs_{ntl_year}_sum', f'viirs_{ntl_year}_median']

    sub_geoquery_cols = [f'viirs_max', f'viirs_median', 'srtm_elevation_500m_na_mean', f'udel_precip_v501_sum_sum', f'ltdr_avhrr_ndvi_v5_yearly_mean', 'dist_to_water_na_mean', 'accessibility_to_cities_2015_v1_0_mean', f'worldpop_pop_count_1km_mosaic_mean', f'esa_landcover_categorical_cropland', 'longitude', 'latitude', f'esa_landcover_categorical_urban', f'esa_landcover_categorical_forest', f'modis_lst_mod11c3_006_day_annual_mean_mean']


    # osm
    all_osm_cols = [i for i in osm_df.columns if i not in dhs_cols]
    sub_osm_cols = [i for i in all_osm_cols if i.startswith(('all_buildings_', 'all_roads_')) and i != 'all_roads_count']

    all_geo_cols = main_geoquery_cols + ['longitude', 'latitude']
    sub_geo_cols = sub_geoquery_cols + ['longitude', 'latitude']

    ntl_cols = [f'viirs_mean', f'viirs_min', f'viirs_max', f'viirs_sum', f'viirs_median']


    project_data_df = all_data_df[dhs_cols + all_osm_cols + main_geoquery_cols]

    project_data_dict[dhs_item] = {
        'all_osm_cols': all_osm_cols,
        'sub_osm_cols': sub_osm_cols,
        'all_geo_cols': all_geo_cols,
        'sub_geo_cols': sub_geo_cols,
        'ntl_cols': ntl_cols,
        'dhs_round': dhs_round,
        'country_utm_epsg_code': country_utm_epsg_code,
        'osm_date': osm_date,
        'geom_id': geom_id,
        'geom_label': geom_label,
        'ntl_year': ntl_year,
        'data': project_data_df
    }





final_data_df = pd.concat([i['data'] for i in project_data_dict.values()], axis=0)

final_data_df.fillna(0, inplace=True)

# for i in final_data.columns:
#     x = final_data[i].isna().sum()
#     if x>0: print(i, x)


final_data_path = os.path.join(data_dir, 'outputs', output_name, 'final_data.csv')
final_data_df.to_csv(final_data_path, index=False)




# -----------------------------------------------------------------------------


def run_model_funcs(data, columns, name, n_splits):

    data_utils.plot_corr(
        data=data,
        features_cols=columns,
        indicator='Wealth Index',
        method='pearsons',
        max_n=50,
        figsize=(10,13),
        output_file=os.path.join(results_dir, f'{name}_pearsons_corr.png'),
        show=show_plots
    )

    data_utils.plot_corr(
        data=data,
        features_cols=columns,
        indicator='Wealth Index',
        method='spearman',
        max_n=50,
        figsize=(10,13),
        output_file=os.path.join(results_dir, f'{name}_spearman_corr.png'),
        show=show_plots
    )

    cv, predictions = model_utils.evaluate_model(
        data=data,
        feature_cols=columns,
        indicator_cols=indicators,
        clust_str=geom_id,
        wandb=None,
        scoring=scoring,
        model_type='random_forest',
        refit='r2',
        search_type=search_type,
        n_splits=n_splits,
        n_iter=10,
        plot_importance=True,
        verbose=2,
        output_file=os.path.join(results_dir, f'{name}_model_cv{n_splits}_'),
        show=show_plots
    )

    model_utils.save_model(cv, data, columns, 'Wealth Index', os.path.join(models_dir, f'{name}_cv{n_splits}_best.joblib'))

    return cv, predictions


# -----------------------------------------------------------------------------
# Explore population distribution and relationships

data_utils.plot_hist(
    final_data_df[f'worldpop_pop_count_1km_mosaic_mean'],
    title='Distribution of Total Population',
    x_label='Total Population',
    y_label='Number of Clusters',
    output_file=os.path.join(results_dir, f'pop_hist.png'),
    show=show_plots
)

pop_r2 = data_utils.plot_regplot(
    final_data_df,
    'Wealth Index',
    'Population',
    f'worldpop_pop_count_1km_mosaic_mean',
    output_file=os.path.join(results_dir, f'pop_wealth_corr.png'),
    show=show_plots
)


# -----------------------------------------------------------------------------

# NTL mean linear model
ntl_r2 = data_utils.plot_regplot(
    data=final_data_df,
    x_label='Wealth Index',
    y_label='Average Nightlight Intensity',
    y_var=f'viirs_mean',
    output_file=os.path.join(results_dir, f'ntl_wealth_regplot.png'),
    show=show_plots
)


# all OSM + NTL
all_osm_ntl_cols = all_osm_cols + ntl_cols
all_osm_ntl_cv, all_osm_ntl_predictions = run_model_funcs(final_data_df, all_osm_ntl_cols, 'all-osm-ntl', n_splits=n_splits)

# NTL only
ntl_cv, ntl_predictions = run_model_funcs(final_data_df, ntl_cols, 'ntl', n_splits=n_splits)

# all OSM only
all_osm_cv, all_osm_predictions = run_model_funcs(final_data_df, all_osm_cols, 'all-osm', n_splits=n_splits)

# all OSM + all geo
all_data_cols = all_osm_cols + all_geo_cols
all_cv, all_predictions = run_model_funcs(final_data_df, all_data_cols, 'all', n_splits=n_splits)

# location only
loc_cv, loc_predictions = run_model_funcs(final_data_df, ['longitude', 'latitude'], 'loc', n_splits=n_splits)

# -----------------

# sub OSM + NTL
sub_osm_ntl_cols = sub_osm_cols + ntl_cols
sub_osm_ntl_cv, sub_osm_ntl_predictions = run_model_funcs(final_data_df, sub_osm_ntl_cols, 'sub-osm-ntl', n_splits=n_splits)

# sub OSM only
sub_osm_cv, sub_osm_predictions = run_model_funcs(final_data_df, sub_osm_cols, 'sub-osm', n_splits=n_splits)

# sub OSM + all geo
sub_osm_all_geo_cols = sub_osm_cols + all_geo_cols
sub_osm_all_geo_cv, sub_osm_all_geo_predictions = run_model_funcs(final_data_df, sub_osm_all_geo_cols, 'sub-osm-all-geo', n_splits=n_splits)

# all geo only
all_geo_cv, all_geo_predictions = run_model_funcs(final_data_df, all_geo_cols, 'all-geo', n_splits=n_splits)

# -----------------

# sub geo
sub_geo_cv, sub_geo_predictions = run_model_funcs(final_data_df, sub_geo_cols, 'sub-geo', n_splits=n_splits)

# sub OSM + sub geo
sub_data_cols = sub_osm_cols + sub_geo_cols
sub_cv, sub_predictions = run_model_funcs(final_data_df, sub_data_cols, 'sub', n_splits=n_splits)

# -----------------

# final set of reduced features
# final_features = ['longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio', 'dist_to_water_na_mean', f'viirs_{ntl_year}_median', f'viirs_{ntl_year}_max', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean']

# final_features = [f'viirs_{ntl_year}_median', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean',  f'viirs_{ntl_year}_max', f'esa_landcover_{ntl_year}_categorical_urban', 'longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio']

final_features = [f'viirs_median', f'worldpop_pop_count_1km_mosaic_mean',  f'viirs_max', f'esa_landcover_categorical_urban', 'longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio']


final_cv, final_predictions = run_model_funcs(final_data_df, final_features, 'final', n_splits=n_splits)


# -----------------------------------------------------------------------------

def print_scores():
    print(f'{project} best r-squared scores ({n_splits}-fold cv):')
    print(f'NTL linear model    & 1     & {round(ntl_r2, 3)} \\\\')
    print(f"Location only       & {loc_cv.n_features_in_}   & {round(loc_cv.best_score_, 3)} \\\\")
    print(f"All OSM + NTL       & {all_osm_ntl_cv.n_features_in_}   & {round(all_osm_ntl_cv.best_score_, 3)} \\\\")
    print(f"NTL                 & {ntl_cv.n_features_in_}   & {round(ntl_cv.best_score_, 3)} \\\\")
    print(f"All OSM             & {all_osm_cv.n_features_in_}   & {round(all_osm_cv.best_score_, 3)} \\\\")
    print(f"All OSM + All geo   & {all_cv.n_features_in_}   & {round(all_cv.best_score_, 3)} \\\\")
    print(f"Sub OSM + NTL       & {sub_osm_ntl_cv.n_features_in_}   & {round(sub_osm_ntl_cv.best_score_, 3)} \\\\")
    print(f"Sub OSM             & {sub_osm_cv.n_features_in_}   & {round(sub_osm_cv.best_score_, 3)} \\\\")
    print(f"Sub OSM + All geo   & {sub_osm_all_geo_cv.n_features_in_}   & {round(sub_osm_all_geo_cv.best_score_, 3)} \\\\")
    print(f"All geo             & {all_geo_cv.n_features_in_}   & {round(all_geo_cv.best_score_, 3)} \\\\")
    print(f"Sub geo             & {sub_geo_cv.n_features_in_}   & {round(sub_geo_cv.best_score_, 3)} \\\\")
    print(f"Sub OSM + Sub geo   & {sub_cv.n_features_in_}   & {round(sub_cv.best_score_, 3)} \\\\")
    print(f"Final               & {final_cv.n_features_in_}   & {round(final_cv.best_score_, 3)} \\\\")

print_scores()

'''

PH_2017_DHS best r-squared scores (5-fold cv):
NTL linear model    & 1     & 0.489 \\
Location only       & 2   & 0.487 \\
All OSM + NTL       & 76   & 0.618 \\
NTL                 & 5   & 0.558 \\
All OSM             & 71   & 0.612 \\
All OSM + All geo   & 104   & 0.667 \\
Sub OSM + NTL       & 11   & 0.57 \\
Sub OSM             & 6   & 0.468 \\
Sub OSM + All geo   & 39   & 0.658 \\
All geo             & 33   & 0.659 \\
Sub geo             & 16   & 0.664 \\
Sub OSM + Sub geo   & 22   & 0.662 \\
Final               & 8   & 0.652 \\


GH_2014_DHS best r-squared scores (5-fold cv):
NTL linear model    & 1     & 0.663 \\
Location only       & 2   & 0.574 \\
All OSM + NTL       & 81   & 0.755 \\
NTL                 & 5   & 0.725 \\
All OSM             & 76   & 0.706 \\
All OSM + All geo   & 108   & 0.85 \\
Sub OSM + NTL       & 11   & 0.761 \\
Sub OSM             & 6   & 0.628 \\
Sub OSM + All geo   & 38   & 0.854 \\
All geo             & 32   & 0.855 \\
Sub geo             & 16   & 0.85 \\
Sub OSM + Sub geo   & 22   & 0.85 \\
Final               & 8   & 0.833 \\
Ecopia              & 8   & 0.843 \\

TG_2013-14_DHS best r-squared scores (5-fold cv):
NTL linear model    & 1     & 0.778 \\
Location only       & 2   & 0.742 \\
All OSM + NTL       & 81   & 0.909 \\
NTL                 & 5   & 0.89 \\
All OSM             & 76   & 0.901 \\
All OSM + All geo   & 109   & 0.93 \\
Sub OSM + NTL       & 11   & 0.886 \\
Sub OSM             & 6   & 0.735 \\
Sub OSM + All geo   & 39   & 0.929 \\
All geo             & 33   & 0.93 \\
Sub geo             & 16   & 0.913 \\
Sub OSM + Sub geo   & 22   & 0.913 \\
Final               & 8   & 0.909 \\

BJ_2017_DHS best r-squared scores (5-fold cv):
NTL linear model    & 1     & 0.75 \\
Location only       & 2   & 0.646 \\
All OSM + NTL       & 75   & 0.781 \\
NTL                 & 5   & 0.781 \\
All OSM             & 70   & 0.762 \\
All OSM + All geo   & 103   & 0.79 \\
Sub OSM + NTL       & 11   & 0.781 \\
Sub OSM             & 6   & 0.645 \\
Sub OSM + All geo   & 39   & 0.79 \\
All geo             & 33   & 0.79 \\
Sub geo             & 16   & 0.788 \\
Sub OSM + Sub geo   & 22   & 0.787 \\
Final               & 8   & 0.781 \\

BJ_2017_DHS best r-squared scores (10-fold cv):
NTL linear model    & 1     & 0.75 \\
Location only       & 2   & 0.643 \\
All OSM + NTL       & 75   & 0.782 \\
NTL                 & 5   & 0.78 \\
All OSM             & 70   & 0.764 \\
All OSM + All geo   & 103   & 0.794 \\
Sub OSM + NTL       & 11   & 0.783 \\
Sub OSM             & 6   & 0.653 \\
Sub OSM + All geo   & 39   & 0.794 \\
All geo             & 33   & 0.792 \\
Sub geo             & 16   & 0.786 \\
Sub OSM + Sub geo   & 22   & 0.788 \\
Final               & 8   & 0.781 \\

KE_2014_DHS best r-squared scores (5-fold cv):
NTL linear model    & 1     & 0.42 \\
Location only       & 2   & 0.557 \\
All OSM + NTL       & 83   & 0.722 \\
NTL                 & 5   & 0.596 \\
All OSM             & 78   & 0.678 \\
All OSM + All geo   & 111   & 0.773 \\
Sub OSM + NTL       & 11   & 0.693 \\
Sub OSM             & 6   & 0.549 \\
Sub OSM + All geo   & 39   & 0.769 \\
All geo             & 33   & 0.767 \\
Sub geo             & 16   & 0.76 \\
Sub OSM + Sub geo   & 22   & 0.765 \\
Final               & 8   & 0.751 \\

    
'''

# -----------------------------------------------------------------------------


if project in ['GH_2014_DHS']:

    ecopia_base_path = base_path = Path(f'/home/userx/Desktop/accessible-poverty-estimates/data/ecopia_data/ghana')

    ecopia_roads_features_path = base_path / f'ghana_{geom_label}_ecopia_roads.csv'
    ecopia_roads_features_df = pd.read_csv(ecopia_roads_features_path)

    ecopia_buildings_features_path = base_path / f'ghana_{geom_label}_ecopia_buildings.csv'
    ecopia_buildings_features_df = pd.read_csv(ecopia_buildings_features_path)

    ecopia_features_df = pd.merge(ecopia_roads_features_df, ecopia_buildings_features_df, on=geom_id)

    ecopia_data_df = pd.merge(ecopia_features_df, final_data_df, on=geom_id)

    ecopia_features = [f'viirs_{ntl_year}_median', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean',  f'viirs_{ntl_year}_max', f'esa_landcover_{ntl_year}_categorical_urban', 'longitude', 'latitude', 'ecopia_roads_length', 'ecopia_buildings_ratio']

    ecopia_cv, ecopia_predictions = run_model_funcs(ecopia_data_df, ecopia_features, 'ecopia', n_splits=n_splits)

    print(f"Ecopia              & {ecopia_cv.n_features_in_}   & {round(ecopia_cv.best_score_, 3)} \\\\")


# -----------------------------------------------------------------------------

exit()


# -----------------------------------------------------------------------------
# exploratory feature reduction

#return a dictionary that identifes a a list of variables for which each variables has a correlation above the specified threshold
#will only return variables that had at least one other variable with the specified correlation threshold
correlated_ivs, corr_matrix = data_utils.corr_finder(final_data_df[all_geo_cols], .8)

for i,j in correlated_ivs.items():
    if j:
        print(i,j)

#subset data based on correlation but make sure that specifically desired covariates are still within the group.
#ensure that even if road/building data can have a low correlation with each other, only one from each group is used per model evaluation
remove_corrs = correlated_ivs['viirs_2016_median'] + correlated_ivs['viirs_2016_max'] + correlated_ivs['worldpop_pop_count_1km_mosaic_2016_mean']  + correlated_ivs['all_roads_length'] + correlated_ivs['all_buildings_ratio']
to_keep = ['viirs_2016_mean','all_buildings_totalarea', 'all_buildings_count', 'all_roads_nearestdist', 'all_roads_length', 'worldpop_pop_count_1km_mosaic_2016_mean']

to_remove_dict = {
    'correlation': [label for label in remove_corrs if label not in to_keep],
    'manual': [],
    # 'osm': [i for i in osm_cols if not i.startswith(('all_buildings_', 'all_roads_'))]
}
to_remove_list = [j for i in to_remove_dict.values() for j in i]

new_features = [i for i in sub_data_cols if i not in to_remove_list]

#create a dataframe based on feature importance the df is ordered from most important to least important feature.
reduction_df = model_utils.rf_feature_importance_dataframe(sub_cv, final_data_df[new_features], final_data_df[indicators])

## subset data based on desired feature importance feature importance
thresh = .004
remove_importance = []
for row, ser in reduction_df.iterrows():
    for idx, val in ser.iteritems():
        if (val < thresh): #if the variable correlates past/at the threshold
            remove_importance.append(row)


important_features = [i for i in new_features if i not in remove_importance]

refined_correlated_ivs, refined_corr_matrix = data_utils.corr_finder(final_data_df[important_features], .65)



# -----------------------------------------------------------------------------
# over-fit comparison for philippines

compare_cols = ['minor_roads_count', 'minor_roads_nearestdist', 'major_roads_nearestdist',  'services_pois_count', 'entertainment_pois_count', 'lodging_pois_count', 'landmark_pois_count', 'viirs.2017.median', 'globalwindatlas_windspeed_na_mean', 'distance_to_gemdata_201708_na_mean', 'dist_to_onshore_petroleum_v12_na_mean', 'distance_to_drugdata_201708_na_mean',  'diamond_distance_201708_na_mean', 'ambient_air_pollution_2013_o3_2013_mean', 'wdpa_iucn_cat_201704_na_count', 'gpw_v4r11_density_2020_mean', 'esa_landcover_2015_categorical_mosaic_cropland']

# distance_to_gemdata_201708 = real
# globalwindatlas_windspeed = real
# dist_to_onshore_petroleum_v12 = real

# distance_to_drugdata_201708 = northwest
# diamond_distance_201708 = southwest
# ambient_air_pollution_2013_o3 = southeast

compare_cv, compare_predictions = model_utils.evaluate_model(
    data=final_data_df,
    feature_cols=compare_cols,
    indicator_cols=indicators,
    clust_str=geom_id,
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    search_type=search_type,
    n_splits=5,
    n_iter=10,
    plot_importance=True,
    verbose=2,
    output_file=os.path.join(results_dir, f'best_compare_model_'),
    show=show_plots
)
