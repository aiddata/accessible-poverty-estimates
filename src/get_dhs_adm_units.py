


import os
import configparser
import glob
from pathlib import Path
import requests

import geopandas as gpd


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

output_name = config[project]['output_name']
dhs_geo_file_name = config[project]['dhs_geo_file_name']

data_dir = project_dir / 'data'


iso3 = 'TGO'


adm1_request = requests.get(f"https://www.geoboundaries.org/gbRequest.html?ISO={iso3}&ADM=ADM1")
adm1_url = adm1_request.json()[0]['gjDownloadURL']
adm1_gdf = gpd.read_file(adm1_url)[['shapeID', 'geometry']]

adm2_request = requests.get(f"https://www.geoboundaries.org/gbRequest.html?ISO={iso3}&ADM=ADM2")
adm2_url = adm2_request.json()[0]['gjDownloadURL']
adm2_gdf = gpd.read_file(adm2_url)[['shapeID', 'geometry']]


def load_dhs_geo(dhs_geo_file_name, data_dir):
    """Load shapefile containing DHS cluster points
    """
    shp_path = glob.glob(str(data_dir / 'dhs' / '**' / dhs_geo_file_name / '*.shp'), recursive=True)[0]
    gdf = gpd.read_file(shp_path)
    # drop locations without coordinates
    gdf = gdf.loc[(gdf["LONGNUM"] != 0) & (gdf["LATNUM"] != 0)].reset_index(drop=True)
    gdf.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"}, inplace=True)
    return gdf

dhs_gdf = load_dhs_geo(dhs_geo_file_name, data_dir)[['DHSID', 'geometry']]

adm1_join = dhs_gdf.sjoin(adm1_gdf, how='left', predicate='intersects').rename(columns={'shapeID': 'ADM1'}).drop(columns=['index_right'])
adm2_join = adm1_join.sjoin(adm2_gdf, how='left', predicate='intersects').rename(columns={'shapeID': 'ADM2'}).drop(columns=['index_right', 'geometry'])

adm2_join.to_csv(str(data_dir / 'dhs' / f'{dhs_geo_file_name}_adm_units.csv'), index=False)
