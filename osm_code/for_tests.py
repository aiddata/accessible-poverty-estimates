# -*- coding: utf-8 -*-

""" Testing space for newly created functions"""

import os
import sys
import pandas as pd
import numpy as np
from joblib import dump, load

import warnings

from scipy.sparse import data
warnings.filterwarnings('ignore')


project_dir = '/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK'  #"/home/userw/Desktop/PHL_WORK"
data_dir = os.path.join(project_dir, 'data')

sys.path.insert(0, os.path.join(project_dir, 'OSM'))

import model_utils
import data_utils


# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


# Scoring metrics
scoring = {
    'r2': data_utils.pearsonr2,
    'rmse': data_utils.rmse
}

# Testing space
#------------------

#Random Datafram
# df = pd.DataFrame(
#     data = {'Column 1': [1,2,3,4,5], 'Column 2': [0,0,0,0,0], 'Column 3': [550,1,250,30,8], 'Column 4': [3,0,0,12,15]}
# ) #create a df with varying levels of correlations 

# true_corr_matrix = df.corr(method='kendall')
# print('Original Matrix:')
# print(true_corr_matrix)

# test_dic, test_corr_matrix = data_utils.corr_finder(df, .5) 
# print('Test dictionary:')
# print(test_dic)
# print('Test Correlation:')
# print(test_corr_matrix)


# OSM + NTL Subsetting: Test
#--------------------------
# osm_ntl = pd.read_csv(os.path.join(data_dir, 'osm_ntl.csv'))
# osm_ntl.set_index('DHSID')

# correlated_ivs, osm_ntl_corr = data_utils.corr_finder(osm_ntl, .6)

# print(osm_ntl_corr.head())

# print(correlated_ivs['ntl_mean'])

# Load models.py data prep
#--------------------

indicators = [
    'Wealth Index'
]


# load in dhs data
final_path =  os.path.join(data_dir, 'dhs_data.csv')
final_df = pd.read_csv(final_path)


# define ntl columns
ntl_cols = [i for i in final_df.columns if i.startswith('ntl_')]


# # OSM data prep

# # original osm data
# osm_roads_file = os.path.join(data_dir, 'osm/features/original_osm_roads.csv')
# osm_buildings_file = os.path.join(data_dir, 'osm/features/original_osm_buildings.csv')
# osm_pois_file = os.path.join(data_dir, 'osm/features/original_osm_pois.csv')
# osm_dhs_id_field = "DHSCLUST"

src_label = "dhs-buffers"

date = "210101"

# new osm data
osm_roads_file = os.path.join(data_dir, 'osm/features/{}_roads_{}.csv'.format(src_label, date))
osm_buildings_file = os.path.join(data_dir, 'osm/features/{}_buildings_{}.csv'.format(src_label, date))
osm_pois_file = os.path.join(data_dir, 'osm/features/{}_pois_{}.csv'.format(src_label, date))
osm_traffic_file = os.path.join(data_dir, 'osm/features/{}_traffic_{}.csv'.format(src_label, date))
osm_transport_file = os.path.join(data_dir, 'osm/features/{}_transport_{}.csv'.format(src_label, date))
osm_dhs_id_field = "DHSID"


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

osm_cols =  [i for i in osm_cols if osm_df[i].min() != osm_df[i].max()]
osm_df = osm_df[[osm_dhs_id_field] + osm_cols]

print("Shape of osm dataframe after drops: {}".format(osm_df.shape))


osm_ntl_cols = osm_cols + ntl_cols

print('Number of OSM + NTL Features:', len(osm_ntl_cols))

# >>>>>
# for use with old osm data
# osm_ntl_df = final_df.merge(osm_df, left_on='Cluster number', right_on=osm_dhs_id_field)

osm_ntl_df = final_df.merge(osm_df, on=osm_dhs_id_field)

osm_ntl_path = os.path.join(data_dir, 'osm_ntl.csv')
osm_ntl_df.to_csv(osm_ntl_path)




# join GeoQuery spatial covars
geoquery_path = os.path.join(data_dir, 'merge_phl_dhs.csv')
geoquery_data = pd.read_csv(geoquery_path)

geoquery_data.fillna(-999, inplace=True)
spatial_df = osm_ntl_df.merge(geoquery_data, on="DHSID", how="left")

# geoquery_cols = [i for i in geoquery_data.columns if len(i.split(".")) == 3 and "categorical" not in i]

geoquery_cols = ['wb_aid.na.sum', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2015.mean', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2016.mean', 'udel_precip_v501_mean.2015.mean', 'udel_precip_v501_mean.2016.mean', 'udel_precip_v501_sum.2015.sum', 'udel_precip_v501_sum.2016.sum', 'udel_air_temp_v501_mean.2015.mean', 'udel_air_temp_v501_mean.2016.mean', 'srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'onshore_petroleum_v12.na.mean', 'oco2_xco2_yearly.2015.mean', 'oco2_xco2_yearly.2016.mean', 'modis_lst_day_yearly_mean.2015.mean', 'modis_lst_day_yearly_mean.2016.mean', 'ltdr_avhrr_ndvi_v5_yearly.2015.mean', 'ltdr_avhrr_ndvi_v5_yearly.2016.mean', 'hansen2018_v16_treecover2000.na.mean', 'gpw_v4_density.2015.mean', 'gpw_v4_count.2015.sum', 'globalwindatlas_windspeed.na.mean', 'globalsolaratlas_pvout.na.mean', 'gemdata_201708.na.sum', 'distance_to_gold_v12.na.mean', 'distance_to_gemdata_201708.na.mean', 'distance_to_drugdata_201708.na.mean', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'dist_to_onshore_petroleum_v12.na.mean', 'dist_to_gadm28_borders.na.mean', 'diamond_distance_201708.na.mean', 'diamond_binary_201708.na.mean', 'ambient_air_pollution_2013_o3.2013.mean', 'ambient_air_pollution_2013_fus_calibrated.2013.mean', 'accessibility_to_cities_2015_v1.0.mean']

geoquery_cols = geoquery_cols + ['wdpa_iucn_cat_201704.na.categorical_count', 'wdpa_iucn_cat_201704.na.categorical_unprotected', 'wdpa_iucn_cat_201704.na.categorical_ia', 'wdpa_iucn_cat_201704.na.categorical_ib','wdpa_iucn_cat_201704.na.categorical_ii', 'wdpa_iucn_cat_201704.na.categorical_iii', 'wdpa_iucn_cat_201704.na.categorical_iv', 'wdpa_iucn_cat_201704.na.categorical_v', 'wdpa_iucn_cat_201704.na.categorical_vi', 'wdpa_iucn_cat_201704.na.categorical_not_applicable', 'wdpa_iucn_cat_201704.na.categorical_not_assigned', 'wdpa_iucn_cat_201704.na.categorical_not_reported', 'wdpa_iucn_cat_201704.na.categorical_mix', 'wdpa_iucn_cat_201704.na.count', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2017.mean', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2018.mean', 'udel_precip_v501_mean.2017.mean', 'udel_precip_v501_sum.2017.sum', 'udel_air_temp_v501_mean.2017.mean', 'oco2_xco2_yearly.2017.mean', 'oco2_xco2_yearly.2018.mean', 'modis_lst_day_yearly_mean.2017.mean', 'ltdr_avhrr_ndvi_v5_yearly.2017.mean', 'ltdr_avhrr_ndvi_v5_yearly.2018.mean', 'hansen2018_v16_lossyear.na.mean', 'gpw_v4_density.2020.mean', 'gpw_v4_count.2020.sum', 'gdp_grid.na.sum', 'esa_landcover_v207.2015.categorical_count', 'esa_landcover_v207.2015.categorical_mosaic_cropland', 'esa_landcover_v207.2015.categorical_rainfed_cropland', 'esa_landcover_v207.2015.categorical_urban', 'esa_landcover_v207.2015.categorical_water_bodies', 'esa_landcover_v207.2015.categorical_forest', 'esa_landcover_v207.2015.categorical_irrigated_cropland', 'esa_landcover_v207.2015.categorical_no_data', 'esa_landcover_v207.2015.categorical_bare_areas', 'esa_landcover_v207.2015.categorical_sparse_vegetation', 'esa_landcover_v207.2015.categorical_grassland', 'esa_landcover_v207.2015.categorical_wetland', 'esa_landcover_v207.2015.categorical_shrubland', 'esa_landcover_v207.2015.categorical_snow_ice', 'drugdata_categorical_201708.na.categorical_count', 'drugdata_categorical_201708.na.categorical_none', 'drugdata_categorical_201708.na.categorical_cannabis', 'drugdata_categorical_201708.na.categorical_coca_bush', 'drugdata_categorical_201708.na.categorical_opium', 'drugdata_categorical_201708.na.categorical_mix', 'categorical_gold_v12.na.categorical_count', 'categorical_gold_v12.na.categorical_none', 'categorical_gold_v12.na.categorical_lootable', 'categorical_gold_v12.na.categorical_surface', 'categorical_gold_v12.na.categorical_nonlootable', 'categorical_gold_v12.na.categorical_mix']


osm_ntl_covar_cols = osm_ntl_cols + geoquery_cols

#Saving entire csv
all_data_path = os.path.join(data_dir, 'osm_ntl_dhs_geo.csv')
spatial_df.to_csv(all_data_path)
#all_data_csv = pd.read_csv(geoquery_path)

print(len(osm_ntl_covar_cols))

# # join DHS spatial covars
# covars_path = os.path.join(data_dir, 'dhs/PH_2017_DHS_03022021_2055_109036/PHGC72FL/PHGC72FL.csv')
# covar_data = pd.read_csv(covars_path)
# spatial_df = osm_ntl_df.merge(covar_data, on="DHSID", how="left")

# covar_cols = covar_data.columns[6:].to_list() + ["pop_sum"]
# osm_ntl_covar_cols = osm_ntl_cols + covar_cols


for i in spatial_df.columns:
    na = spatial_df[i].isna().sum()
    if na > 0: print(i, spatial_df[i].isna().sum())



#---------------------------------------------------------

correlated_ivs, corr_matrix = data_utils.corr_finder(spatial_df, .85) #will only return variables that had at least one other variable with an .85 correlation


updated_df, new_features = data_utils.subset_dataframe(spatial_df, osm_ntl_covar_cols, correlated_ivs['ntl_mean'])



test1_cv, test1_predictions = model_utils.evaluate_model(
    data=updated_df2,
    feature_cols=new_features2,
    indicator_cols=indicators,
    search_type="grid",  ## this is a change from the initial 
    clust_str="Cluster number",
    wandb=None,
    scoring=scoring,
    model_type='random_forest',
    refit='r2',
    n_splits=4,
    n_iter=10,
    plot_importance=True,
    verbose=2
)


# #define X,y for all data
# test1_X = new_df[new_df_cols]
# test1_y = new_df['Wealth Index'].tolist()


# # refit cv model with all data
# test1_best = test1_cv.best_estimator_.fit(test1_X, test1_y)

# # save model
# test1_model_path = os.path.join(data_dir, 'models/test1_best.joblib')
# dump(test1_best, test1_model_path)

# print(updated_df.shape)