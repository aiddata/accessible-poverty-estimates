"""
python 3.9

portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Prepare DHS data

- download DHS data (requires registration) into ${project_dir}/data/dhs
    - https://dhsprogram.com/data/dataset/Philippines_Standard-DHS_2017.cfm?flag=1
    - 2017 household recode and gps (PHHR71DT.ZIP, PHGE71FL.ZIP)
- unzip both files
"""

import os
import configparser
import glob
from pathlib import Path

import pandas as pd
import geopandas as gpd


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

dhs_round = config[project]['dhs_round']
dhs_hh_file_name = config[project]['dhs_hh_file_name']
dhs_geo_file_name = config[project]['dhs_geo_file_name']
country_utm_epsg_code = config[project]['country_utm_epsg_code']


data_dir = os.path.join(project_dir, 'data')

os.makedirs(os.path.join(data_dir, 'outputs', dhs_round), exist_ok=True)


# ---------------------------------------------------------
# create extract job file to use with resulting dhs buffers geojson for geoquery extract

iso2 = project.lower().split('_')[0]
base_extract_job_path = Path(data_dir, 'extract_job.json')
extract_job_path = Path(data_dir, 'outputs', dhs_round, 'extract_job.json')
extract_job_text = base_extract_job_path.read_text().replace('[[ISO2]]', iso2).replace('[[DHS_ROUND]]', dhs_round)
extract_job_path.write_text(extract_job_text)


# ---------------------------------------------------------
# prepare DHS cluster indicators

dhs_file = glob.glob(os.path.join(data_dir, 'dhs', f'{dhs_round}*', dhs_hh_file_name, '*.DTA' ))[0]
dhs_dict_file = glob.glob(os.path.join(data_dir, 'dhs', f'{dhs_round}*', dhs_hh_file_name, '*.DO' ))[0]

dhs = pd.read_stata(dhs_file, convert_categoricals=False)

"""
Column      - Description
hv001       - Cluster number
hv271       - Wealth index factor score combined (5 decimals)
hv108_01    - Education completed in single years
hv206       - Has electricity
hv204       - Access to water (minutes)
hv226       - Type of cooking fuel
hv208       - Has television
"""

var_titles = {

}

var_list = ["hv001", "hv271"]
hh_data = dhs[var_list].copy(deep=True)

# 996 = water is on property
if "hv204" in var_list:
    hh_data["hv204"] = hh_data["hv204"].replace(996, 0)


agg_rules = {
    "hv271": "mean",
    "hv108_01": "mean",
    "hv206": "mean",
    "hv204": "median",
    "hv208": "mean",
    "hv226": "mean"
}

active_agg_rules = {i:j for i,j in agg_rules.items() if i in var_list}

cluster_data = hh_data.groupby("hv001").agg(active_agg_rules).reset_index().dropna(axis=1)

print('Cluster Data Dimensions: {}'.format(cluster_data.shape))

# cluster_data.columns = ["cluster_id", "wealthscore", "education_years", "electricity", "time_to_water"]

cluster_data.columns = [
    'Cluster number',
    'Wealth Index',
    # 'Education completed (years)',
    # 'Access to electricity',
    # 'Access to water (minutes)'
    # 'Has television',
    # 'Type of cooking fuel'
]


#  ---------------------------------------------------------
# read in IR file along with the label file
# v001        - Cluster number
# v171b       - Internet usage in the last month

# dhs_ir_file = os.path.join(data_dir, 'dhs/PHIR71DT/PHIR71FL.DTA')
# dhs_ir_dict_file = os.path.join(data_dir,  'dhs/PHHR71DT/PHIR71FL.DO')

# dhs_ir = pd.read_stata(dhs_ir_file, convert_categoricals=False)
# print(dhs_ir.shape)

# ir_data = dhs_ir[["v001","v171b"]]

# ir_data = ir_data.groupby("v001").agg({
#     "v171b": "mean",
# }).reset_index().dropna(axis=1)

# ir_data.columns = [
#     'Cluster number',
#     'Internet usage in the last month'
# ]

# ir_out_path = os.path.join(data_dir, "ir_indicators.csv")
# ir_data.to_csv(ir_out_path, encoding="utf-8")


# # ---------------------------------------------------------
# prepare DHS cluster geometries

shp_path = glob.glob(os.path.join(data_dir, 'dhs', f'{dhs_round}*', dhs_geo_file_name, '*.shp' ))[0]
raw_gdf = gpd.read_file(shp_path)

# drop locations without coordinates
raw_gdf = raw_gdf.loc[(raw_gdf["LONGNUM"] != 0) & (raw_gdf["LATNUM"] != 0)].reset_index(drop=True)

raw_gdf.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"}, inplace=True)

def buffer(geom, urban_rural):
    """Create DHS cluster buffers

    Buffer size:
        - 2km for urban
        - 5km for rural (1% of rural have 10km displacement, but ignoring those)

    Metric units converted to decimal degrees by dividing by width of one decimal
    degree in km at equator. Not an ideal buffer created after reprojecting,
    but good enough for this application.
    """
    if urban_rural == "U":
        return geom.buffer(2000)
    elif urban_rural == "R":
        return geom.buffer(5000)
    else:
        raise ValueError("Invalid urban/rural identified ({})".format(urban_rural))


gdf = raw_gdf.copy(deep=True)
# convert to UTM first (meters) then back to WGS84 (degrees)
gdf = gdf.to_crs(f"EPSG:{country_utm_epsg_code}")
# generate buffer
gdf.geometry = gdf.apply(lambda x: buffer(x.geometry, x.URBAN_RURA), axis=1)
gdf = gdf.to_crs("EPSG:4326") # WGS84


# ---------------------------------------------------------
# prepare combined dhs cluster data

# merge geospatial data with dhs hr indicators
gdf_merge = gdf.merge(cluster_data, left_on="DHSCLUST", right_on="Cluster number", how="inner")


# output all dhs cluster data to CSV
final_df = gdf_merge[[i for i in gdf_merge.columns if i != "geometry"]]
final_path =  os.path.join(data_dir, 'outputs', dhs_round, 'dhs_data.csv')
final_df.to_csv(final_path, index=False, encoding='utf-8')


# save buffer geometry as geojson
geo_path = os.path.join(data_dir, 'outputs', dhs_round, 'dhs_buffers.geojson')
gdf[['DHSID', 'geometry']].to_file(geo_path, driver='GeoJSON')

