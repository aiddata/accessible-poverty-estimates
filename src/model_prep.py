"""
python 3.9

portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Run models based on OSM features and additional geospatial data


"""

import os
import configparser
import json
from pathlib import Path

import pandas as pd



if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

primary_geom_id = config[project]["geom_id"]

indicators = json.loads(config["main"]['indicators'])
# indicators = [
#     'Wealth Index',
#     'Education completed (years)',
#     'Access to electricity',
#     'Access to water (minutes)'
# ]


data_dir = os.path.join(project_dir, 'data')

output_name = config[project]['output_name']

outut_dir = os.path.join(data_dir, 'outputs', output_name)

os.makedirs(outut_dir, exist_ok=True)




if 'combination' in config[project] and config[project]['combination'] == 'True':
    dhs_list = config[project]['project_list'].replace(' ', '').split(',')

else:
    dhs_list = [project]



project_data_dict = {}

for dhs_item in dhs_list:

    tmp_output_name = config[dhs_item]['output_name']
    country_utm_epsg_code = config[dhs_item]['country_utm_epsg_code']

    osm_date = config[dhs_item]["osm_date"]
    geom_id = config[dhs_item]["geom_id"]
    geom_label = config[dhs_item]["geom_label"]
    dhs_geo_file_name = config[dhs_item]['dhs_geo_file_name']

    geoquery_data_file_name = config[dhs_item]["geoquery_data_file_name"]

    ntl_year = config[dhs_item]["ntl_year"]

    geospatial_variable_years = json.loads(config[dhs_item]['geospatial_variable_years'])


    # -------------------------------------
    # load in dhs data


    adm_path =  os.path.join(data_dir, 'dhs', f'{dhs_geo_file_name}_adm_units.csv')
    adm_df = pd.read_csv(adm_path)

    dhs_path =  os.path.join(data_dir, 'outputs', tmp_output_name, 'dhs_data.csv')
    raw_dhs_df = pd.read_csv(dhs_path)

    raw_dhs_df = raw_dhs_df.merge(adm_df, on=geom_id, how='left')

    dhs_cols = [geom_id, 'latitude', 'longitude'] + indicators + ['ADM1','ADM2']

    dhs_df = raw_dhs_df[dhs_cols]


    # -------------------------------------
    # OSM data prep

    osm_features_dir = os.path.join(data_dir, 'outputs', tmp_output_name, 'osm_features')

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
    geoquery_path = os.path.join(data_dir, 'outputs', tmp_output_name, f'{geoquery_data_file_name}.csv')
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

    project_data_df.rename(columns={geom_id: primary_geom_id}, inplace=True)

    project_data_dict[dhs_item] = {
        'all_osm_cols': all_osm_cols,
        'sub_osm_cols': sub_osm_cols,
        'all_geo_cols': all_geo_cols,
        'sub_geo_cols': sub_geo_cols,
        'ntl_cols': ntl_cols,
        'country_utm_epsg_code': country_utm_epsg_code,
        'osm_date': osm_date,
        'geom_id': geom_id,
        'geom_label': geom_label,
        'ntl_year': ntl_year,
        'data': project_data_df
    }



all_osm_cols = list(set.union(*[set(i['all_osm_cols']) for i in project_data_dict.values()]))
sub_osm_cols = list(set.union(*[set(i['sub_osm_cols']) for i in project_data_dict.values()]))
all_geo_cols = list(set.union(*[set(i['all_geo_cols']) for i in project_data_dict.values()]))
sub_geo_cols = list(set.union(*[set(i['sub_geo_cols']) for i in project_data_dict.values()]))
ntl_cols = list(set.union(*[set(i['ntl_cols']) for i in project_data_dict.values()]))

final_data_df = pd.concat([i['data'] for i in project_data_dict.values()], axis=0)

final_data_df.fillna(0, inplace=True)

# for i in final_data.columns:
#     x = final_data[i].isna().sum()
#     if x>0: print(i, x)


final_data_path = os.path.join(data_dir, 'outputs', output_name, 'final_data.csv')
final_data_df.to_csv(final_data_path, index=False)


json_data = {
    'all_osm_cols': all_osm_cols,
    'sub_osm_cols': sub_osm_cols,
    'all_geo_cols': all_geo_cols,
    'sub_geo_cols': sub_geo_cols,
    'ntl_cols': ntl_cols,
    'primary_geom_id': primary_geom_id,
}

json_path = os.path.join(data_dir, 'outputs', output_name, 'final_data.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

