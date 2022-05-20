'''
Script to assist with download OSM data from GeoFabrik
- also converts buildings/roads OSM data to SpatiaLite databases

Note: this uses the project config.ini file to set data directory, but this script does NOT use
      project specific information (e.g. country name, osm date, etc.) to determine what to download.
      See details on defining countries below.

Set desired date of OSM data
    - 'latest' for latest available
    - monthly data is typically available for current year, and data for january 1 from previous years
        - data following these guidelines may not always exist

Update country dict with the countries to be downloaded for each region (based on GeoFabrik naming, see their website for exact region/country names)

'''

import os
import configparser
import subprocess as sp
import requests
import hashlib
import zipfile
from pathlib import Path


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

data_dir = Path(project_dir) / 'data'

output_dir = data_dir / 'osm'

osm_date = '220101'

# key is region, value is list of country names in region
country_dict = {
    'africa': ['Liberia', 'Sierra Leone', 'Cameroon', 'Guinea', 'Mali', 'Nigeria', 'Zambia', 'Benin', 'Burundi', 'Ethiopia', 'South Africa', 'Uganda', 'Angola', 'Malawi', 'Rwanda', 'Tanzania', 'Zimbabwe', 'Chad', 'Ghana', 'Kenya', 'Lesotho'],
    'antarctica': [],
    'asia': [],
    'australia': [],
    'central-america': [],
    'europe': [],
    'north-america': [],
    'south-america': []
}

download_list = []
for region, country_names in country_dict.items():
    download_list.extend([(region, i.lower().replace(' ', '-')) for i in country_names])

base_url = 'http://download.geofabrik.de'

def download_file(url, path, overwrite=False):
    if not path.exists() or overwrite:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return path


def download_and_verify(dl_url, dl_path, true_md5):
    valid = False
    overwrite = False
    attempt = 0
    max_attempts = 3
    while not valid and attempt < max_attempts:
        download_file(dl_url, dl_path, overwrite=overwrite)
        overwrite = True
        dl_md5 = hashlib.md5(open(dl_path,'rb').read()).hexdigest()
        valid = dl_md5 == true_md5
    return valid


for region, country_name in download_list:

    fname = f'{country_name}-{osm_date}-free.shp'
    print(fname)

    shp_path = output_dir / f'{fname}.zip'
    shp_url = f'{base_url}/{region}/{fname}.zip'

    md5_url = f'{base_url}/{region}/{fname}.zip.md5'
    md5_path = output_dir / f'{fname}.zip.md5'

    print('\tDownloading and/or verifying...')

    _ = download_file(md5_url, md5_path)
    true_md5 = open(md5_path,'rb').read().decode("utf-8").split(' ')[0]

    valid = download_and_verify(shp_url, shp_path, true_md5)
    if not valid:
        print(f'{fname} - failed to download and verify')

    print('\tUnzipping...')

    unzip_dir = output_dir / fname
    if not unzip_dir.exists():
        with zipfile.ZipFile(shp_path, 'r') as zip:
            zip.extractall(unzip_dir)


    osm_buildings_sqlite_path = data_dir / 'osm' / f'{country_name}-{osm_date}-free.shp' /'gis_osm_buildings_a_free_1.sqlite'

    if not osm_buildings_sqlite_path.exists():

        osm_buildings_shp_path = data_dir / 'osm' / f'{country_name}-{osm_date}-free.shp' /'gis_osm_buildings_a_free_1.shp'

        building_table_name = 'DATA_TABLE'

        buildings_call_str = f'ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln {building_table_name} -dsco SPATIALITE=YES {osm_buildings_sqlite_path} {osm_buildings_shp_path}'

        print('\tConverting buildings to spatialite...')
        buildings_call = sp.run(buildings_call_str, shell=True, capture_output=True)

        if buildings_call.returncode != 0:
            raise Exception(buildings_call.stderr, buildings_call_str)


    osm_roads_sqlite_path = data_dir / 'osm' / f'{country_name}-{osm_date}-free.shp' / 'gis_osm_roads_free_1.sqlite'

    if not osm_roads_sqlite_path.exists():

        osm_roads_shp_path = data_dir / 'osm' / f'{country_name}-{osm_date}-free.shp' / 'gis_osm_roads_free_1.shp'

        road_table_name = 'DATA_TABLE'

        roads_call_str = f'ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln {road_table_name} -dsco SPATIALITE=YES {osm_roads_sqlite_path} {osm_roads_shp_path}'

        print('\tConverting roads to spatialite...')
        roads_call = sp.run(roads_call_str, shell=True, capture_output=True)

        if roads_call.returncode != 0:
            raise Exception(roads_call.stderr, roads_call_str)

