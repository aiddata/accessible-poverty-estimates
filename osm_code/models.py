"""
python 3.8

portions of code and/or methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Run models based on OSM features and additional geospatial data



"""

import os
import sys
import pandas as pd
import numpy as np
from joblib import dump, load

import warnings
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


# define ntl columns
ntl_cols = [i for i in final_df.columns if i.startswith('ntl_')]



# -------------------------------------
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


# -------------------------------------
#Additional spatial covar data prep           #NOTE to Sasan: RUN geoquery spatial covers after initial analysis


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

##------------------------------------
#population subsetting 


# # -----------------------------------------------------------------------------
# Explore population distribution


# data_utils.plot_hist(
#     final_df['pop_sum'],
#     title='Distribution of Total Population',
#     x_label='Total Population',
#     y_label='Number of Clusters'
# )

# data_utils.plot_regplot(final_df, 'Wealth Index', 'Population', 'pop_sum')
# data_utils.plot_regplot(final_df, 'Wealth Index', y_var="ntl_max")
# data_utils.plot_regplot(final_df, 'Wealth Index')
# data_utils.plot_regplot(final_df, 'Education completed (years)')
# data_utils.plot_regplot(final_df, 'Access to electricity')
# data_utils.plot_regplot(final_df, 'Access to water (minutes)')



# # -----------------------------------------------------------------------------
# NTL only models

# data_utils.plot_corr(
#     data=final_df,
#     features_cols=ntl_cols,
#     indicator='Wealth Index',
#     max_n=4,
#     figsize=(8,6)
# )
# #

# predictions = model_utils.evaluate_model(
#     data=final_df,
#     feature_cols=ntl_cols,
#     indicator_cols=indicators,
#     scoring=scoring,
#     model_type='random_forest',
#     refit='r2',
#     search_type='random',
#     n_splits=5,
#     n_iter=10,
#     plot_importance=True,
#     verbose=2
# )

# predictions = model_utils.evaluate_model(
#     data=final_df,
#     feature_cols=ntl_cols,
#     indicator_cols=indicators,
#     scoring=scoring,
#     model_type='xgboost',
#     refit='r2',
#     search_type='random',
#     n_splits=5,
#     n_iter=10
# )

# predictions = model_utils.evaluate_model(
#     data=final_df,
#     feature_cols=ntl_cols,
#     indicator_cols=indicators,
#     scoring=scoring,
#     model_type='lasso',
#     refit='r2',
#     search_type='grid',
#     n_splits=5
# )

# predictions = model_utils.evaluate_model(
#     data=final_df,
#     feature_cols=ntl_cols,
#     indicator_cols=indicators,
#     scoring=scoring,
#     model_type='ridge',
#     refit='r2',
#     search_type='grid',
#     n_splits=5
# )



# # -----------------------------------------------------------------------------
# OSM + NTL models

# data_utils.plot_corr(
#     data=osm_ntl_df,
#     features_cols=osm_cols,
#     indicator='Wealth Index',
#     max_n=50,
#     figsize=(10,13)
# )


# osm_cv, osm_predictions = model_utils.evaluate_model(
#     data=osm_ntl_df,
#     feature_cols=osm_ntl_cols,
#     indicator_cols=indicators,
#     clust_str="Cluster number",
#     wandb=None,
#     scoring=scoring,
#     model_type='random_forest',
#     refit='r2',
#     search_type='random',
#     n_splits=5,
#     n_iter=10,
#     plot_importance=True,
#     verbose=2
# )


# # define X,y for all data
# osm_X = osm_ntl_df[osm_ntl_cols]
# osm_y = osm_ntl_df['Wealth Index'].tolist()


# # refit cv model with all data
# osm_best = osm_cv.best_estimator_.fit(osm_X, osm_y)

# # save model
# osm_model_path = os.path.join(data_dir, 'models/osm_ntl_best.joblib') #added ntl to list
# dump(osm_best, osm_model_path)


# -----------------------------------------------------------------------------
# NTL + OSM + spatial models

data_utils.plot_corr(
    data=spatial_df,
    features_cols=geoquery_cols,
    indicator='Wealth Index',
    max_n=50,
    figsize=(10,13),
    name='all_variables',
    pathway = '/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK/data/models/correlation_plots'
)

test1_cv, test1_predictions = model_utils.evaluate_model(
    data=spatial_df,
    feature_cols=osm_ntl_covar_cols,
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


# define X,y for all data
test1_X = spatial_df[osm_ntl_covar_cols]
test1_y = spatial_df['Wealth Index'].tolist()


# refit cv model with all data
test1_best = test1_cv.best_estimator_.fit(test1_X, test1_y)

# save model
test1_model_path = os.path.join(data_dir, 'models/test1_best.joblib')
dump(test1_best, test1_model_path)



# # -----------------------------------------------------------------------------
# # load KC data

# # load in main kc data
# kc_path =  os.path.join(data_dir, 'kc-clusters_ntl_pop.csv')
# kc_df = pd.read_csv(kc_path)

# kc_df.rename(columns={"cluster": "cluster_name"}, inplace=True)

# # define subset of kc ntl extract columns to match dhs ntl extract cols

# kc_buffer_size = "1km"

# # kc_df.rename(columns={i:"{0}_{2}".format(*i.split("_")) for i in kc_df.columns if i.startswith('ntl_{}_'.format(kc_buffer_size))}, inplace=True)

# for i in kc_df.columns:
#     if i.startswith('ntl_{}_'.format(kc_buffer_size)):
#         kc_df["ntl_{}".format(i.split("_")[2])] = kc_df[i]

# kc_ntl_cols = ['ntl_min', 'ntl_max', 'ntl_mean', 'ntl_sum', 'ntl_median']

# # -------------------------------------
# # OSM data prep


# kc_id_field = "cluster_name"

# src_label = "kc-{}-buffers".format(kc_buffer_size)

# date = "210101"

# # new osm data
# kc_osm_roads_file = os.path.join(data_dir, 'osm/features/{}_roads_{}.csv'.format(src_label, date))
# kc_osm_buildings_file = os.path.join(data_dir, 'osm/features/{}_buildings_{}.csv'.format(src_label, date))
# kc_osm_pois_file = os.path.join(data_dir, 'osm/features/{}_pois_{}.csv'.format(src_label, date))
# kc_osm_traffic_file = os.path.join(data_dir, 'osm/features/{}_traffic_{}.csv'.format(src_label, date))
# kc_osm_transport_file = os.path.join(data_dir, 'osm/features/{}_transport_{}.csv'.format(src_label, date))



# # Load OSM datasets
# kc_roads = pd.read_csv(kc_osm_roads_file)
# kc_buildings = pd.read_csv(kc_osm_buildings_file)
# kc_pois = pd.read_csv(kc_osm_pois_file)
# kc_traffic = pd.read_csv(kc_osm_traffic_file)
# kc_transport = pd.read_csv(kc_osm_transport_file)

# kc_osm_df_list = [kc_roads, kc_buildings, kc_pois, kc_traffic, kc_transport]



# kc_osm_df = pd.concat(kc_osm_df_list, join="inner", axis=1)


# kc_osm_df = kc_osm_df.loc[:, ~kc_osm_df.columns.duplicated()]

# kc_osm_df = kc_osm_df[[i for i in kc_osm_df.columns if "_roads_nearest-osmid" not in i]]


# print("Shape of osm dataframe: {}".format(kc_osm_df.shape))

# # Get list of columns
# kc_osm_cols = list(kc_osm_df.columns[1:])

# # kc_osm_cols =  [i for i in kc_osm_cols if kc_osm_df[i].min() != kc_osm_df[i].max()]
# kc_osm_df = kc_osm_df[[kc_id_field] + kc_osm_cols]

# print("Shape of osm dataframe after drops: {}".format(kc_osm_df.shape))


# kc_osm_ntl_cols = kc_osm_cols + kc_ntl_cols

# kc_df.rename(columns={"cluster": "cluster_name"}, inplace=True)

# kc_osm_ntl_df = kc_df.merge(kc_osm_df, on=kc_id_field)


# # -------------------------------------
# # Additional spatial covar data prep


# # join GeoQuery spatial covars
# kc_geoquery_path = os.path.join(data_dir, 'merge_phl_kc_{}.csv'.format(kc_buffer_size))
# kc_geoquery_data = pd.read_csv(kc_geoquery_path)
# kc_geoquery_data.rename(columns={"cluster": "cluster_name"}, inplace=True)

# kc_geoquery_data.fillna(-999, inplace=True)
# kc_spatial_df = kc_osm_ntl_df.merge(kc_geoquery_data, on=kc_id_field, how="left")

# # geoquery_cols = [i for i in geoquery_data.columns if len(i.split(".")) == 3 and "categorical" not in i]

# kc_geoquery_cols = ['wb_aid.na.sum', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2015.mean', 'viirs_vcmcfg_dnb_composites_v10_yearly_max.2016.mean', 'udel_precip_v501_mean.2015.mean', 'udel_precip_v501_mean.2016.mean', 'udel_precip_v501_sum.2015.sum', 'udel_precip_v501_sum.2016.sum', 'udel_air_temp_v501_mean.2015.mean', 'udel_air_temp_v501_mean.2016.mean', 'srtm_slope_500m.na.mean', 'srtm_elevation_500m.na.mean', 'onshore_petroleum_v12.na.mean', 'oco2_xco2_yearly.2015.mean', 'oco2_xco2_yearly.2016.mean', 'modis_lst_day_yearly_mean.2015.mean', 'modis_lst_day_yearly_mean.2016.mean', 'ltdr_avhrr_ndvi_v5_yearly.2015.mean', 'ltdr_avhrr_ndvi_v5_yearly.2016.mean', 'hansen2018_v16_treecover2000.na.mean', 'gpw_v4_density.2015.mean', 'gpw_v4_count.2015.sum', 'globalwindatlas_windspeed.na.mean', 'globalsolaratlas_pvout.na.mean', 'gemdata_201708.na.sum', 'distance_to_gold_v12.na.mean', 'distance_to_gemdata_201708.na.mean', 'distance_to_drugdata_201708.na.mean', 'distance_to_coast_236.na.mean', 'dist_to_water.na.mean', 'dist_to_onshore_petroleum_v12.na.mean', 'dist_to_gadm28_borders.na.mean', 'diamond_distance_201708.na.mean', 'diamond_binary_201708.na.mean', 'ambient_air_pollution_2013_o3.2013.mean', 'ambient_air_pollution_2013_fus_calibrated.2013.mean', 'accessibility_to_cities_2015_v1.0.mean']

# kc_osm_ntl_covar_cols = kc_osm_ntl_cols + kc_geoquery_cols






# data_utils.plot_regplot(kc_df, 'ind1_4_ha', y_var="ntl_mean")


# data_utils.plot_corr(
#     data=kc_osm_ntl_df,
#     features_cols=kc_osm_ntl_cols,
#     indicator='ind1_3_ha',
#     max_n=20,
#     figsize=(8,6)
# )


# kc_all_cv, kc_all_predictions = model_utils.evaluate_model(
#     data=kc_osm_ntl_df,
#     feature_cols=kc_osm_ntl_cols,
#     indicator_cols=["ind1_3_ha"],
#     clust_str="cluster_name",
#     wandb=None,
#     scoring=scoring,
#     model_type='random_forest',
#     refit='r2',
#     search_type='random',
#     n_splits=3,
#     n_iter=10,
#     plot_importance=True,
#     verbose=2
# )


# # -------------------------------------


# kc_X = kc_spatial_df[kc_osm_ntl_covar_cols]

# kc_final_cols = all_X.columns.to_list()
# kc_X = kc_spatial_df[kc_final_cols]






# # missing_cols = [i for i in X.columns if i not in kc_X.columns]

# # extra_cols = [i for i in kc_X.columns if i not in X.columns]

# kc_X = kc_osm_ntl_df[kc_osm_ntl_cols]

# kc_final_cols = osm_X.columns.to_list()
# kc_X = kc_osm_ntl_df[kc_final_cols]



# # -------------------------------------

# # load model
# model_path = os.path.join(data_dir, 'models/osm_best.joblib')
# # model_path = os.path.join(data_dir, 'models/all_best.joblib')
# best = load(model_path)


# # refit cv model with all data
# kc_pred_awi = best.predict(kc_X)


# # -------------------------------------
# # -------------------------------------

# eval_df = kc_osm_ntl_df.copy(deep=True)
# eval_df["pred_awi"] = kc_pred_awi

# eval_df["any_ntl"] = eval_df["ntl_mean"] > 0

# for j in ["buildings", "roads", "traffic", "transport", "pois"]:
#     eval_df["total_{}".format(j)] = eval_df[[i for i in eval_df.columns if "_{}_count".format(j) in i]].sum(axis=1)
#     eval_df["any_{}".format(j)] = eval_df["total_{}".format(j)] > 0



# eval_df.to_csv("/home/userw/Desktop/kc_eval.csv", encoding='utf-8', sep=',', index=False)

# # -------------------------------------
# # -------------------------------------

# # gen endline correlation of prediction vs KC assets

# kc_assets = kc_df["ind1_4"] # consumption
# kc_assets = kc_df["ind1_3_ha"] # assets pca
# kc_assets = kc_df["ind1_4_ha"] # assets php


# import sklearn
# import scipy
# import seaborn as sns
# import matplotlib.pyplot as plt

# tmp_x = sklearn.preprocessing.minmax_scale(kc_df["ind1_3_ha"], feature_range=(0, 1), axis=0, copy=True)
# tmp_y = sklearn.preprocessing.minmax_scale(kc_pred_awi, feature_range=(0, 1), axis=0, copy=True)

# tmp_x = kc_df["ind1_3_ha"]
# tmp_y = kc_pred_awi
# ax = sns.regplot(
#     tmp_x,
#     tmp_y,
#     color="g"
# )
# plt.title('pearsons r2 = {}'.format(round(scipy.stats.pearsonr(tmp_x, tmp_y)[0], 3)))
# plt.xlabel("assets pca" )
# plt.ylabel("pred awi")
# plt.show()



# # -------------------------------------
# # -------------------------------------
# # correlation between DHS cluster wealth index and extract RWI at DHS buffers

# dhs_rwi_path = "/home/userw/Desktop/phl_tmp/dhs_rwi.csv"

# dhs_rwi_df = pd.read_csv(dhs_rwi_path)

# dhs_rwi_df["rwi"] = dhs_rwi_df["rwi_1"]

# dhs_rwi_df = dhs_rwi_df.loc[~dhs_rwi_df.rwi.isna()]

# x = "Wealth Index"
# # x = "ntl_max"
# y = "rwi_1"

# ax = sns.regplot(
#     dhs_rwi_df[x],
#     dhs_rwi_df[y],
#     color="g"
# )
# plt.title('pearsons r2 = {}'.format(round(scipy.stats.pearsonr(dhs_rwi_df[x], dhs_rwi_df[y])[0], 3)))
# plt.xlabel(x)
# plt.ylabel(y)
# plt.show()




# # -------------------------------------
# # -------------------------------------
# # correlation between KC cluster endline, rwi, and pred awi
# #   repeat correlation with filter based on distance from kc to nearest dhs (based on dhs buffer verticle)



# import os
# import sys
# import pandas as pd
# import numpy as np
# from joblib import dump, load
# import geopandas as gpd
# from shapely.geometry import Point
# from sklearn.neighbors import BallTree

# import scipy
# import seaborn as sns
# import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('ignore')


# project_dir = "/home/userw/Desktop/PHL_WORK"
# data_dir = os.path.join(project_dir, 'data')

# sys.path.insert(0, os.path.join(project_dir, 'OSM'))

# import model_utils
# import data_utils



# rwi_path = "/home/userw/Desktop/phl_tmp/kc_eval_rwi.csv"
# rwi_df = pd.read_csv(rwi_path)
# rwi_df["rwi"] = rwi_df["rwi_1"]

# # rwi_path = "/home/userw/Desktop/phl_tmp/kc_hh_rwi.csv"
# # rwi_df = pd.read_csv(rwi_path)
# # rwi_df["rwi"] = rwi_df["1"]

# # rwi_df.rwi.fillna(0, inplace=True)
# rwi_df = rwi_df.loc[~rwi_df.rwi.isna()]

# # get kc locations
# dist_df = rwi_df.copy(deep=True)
# dist_df["geometry"] = dist_df.apply(lambda x: Point(x.cluster_lon, x.cluster_lat), axis=1)
# dist_df = gpd.GeoDataFrame(dist_df)

# src_points = dist_df.apply(lambda x: (x.cluster_lon, x.cluster_lat), axis=1).to_list()


# # load dhs buffers
# dhs_buffers_path = os.path.join(data_dir, 'dhs_buffers.geojson')
# dhs_buffers = gpd.read_file(dhs_buffers_path)
# feat_id_field = "DHSID"
# # generate list of all dhs vertices and convert to geodataframe
# line_xy = dhs_buffers.apply(lambda x: (x[feat_id_field], x.geometry.exterior.coords.xy), axis=1)
# line_xy_lookup = [j for i in line_xy for j in list(zip([i[0]]*len(i[1][0]), *i[1]))]
# line_xy_df = pd.DataFrame(line_xy_lookup, columns=[feat_id_field, "x", "y"])
# line_xy_points = [(i[1], i[2]) for i in line_xy_lookup]
# # create ball tree for nearest point lookup
# #  see https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
# tree = BallTree(line_xy_points, leaf_size=50, metric='haversine')
# # query tree
# distances, indices = tree.query(src_points, k=1)
# distances = distances.transpose()
# indices = indices.transpose()
# # k=1 so output length is array of len=1
# closest = indices[0]
# closest_dist = distances[0]
# # def func to get id for closest locations
# id_lookup = lambda idx: line_xy_df.loc[idx][feat_id_field]
# # set final data
# dist_df["nearest-id"] = list(map(id_lookup, closest))
# dist_df["nearest-dist"] = closest_dist *111123


# comparison_df = dist_df.loc[dist_df["nearest-dist"] < 1000].copy(deep=True)

# x = "ind1_3_ha"
# # x = "ind1_4_ha"
# # x = "ntl_1km_mean"
# # x = "pred_awi"
# y = "rwi"

# ax = sns.regplot(
#     comparison_df[x],
#     comparison_df[y],
#     color="g"
# )
# plt.title('pearsons r2 = {}'.format(round(scipy.stats.pearsonr(comparison_df[x], comparison_df[y])[0], 3)))
# plt.xlabel(x)
# plt.ylabel(y)
# plt.show()

