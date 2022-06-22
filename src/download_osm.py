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

import prefect
from prefect import task, Flow, Client
from prefect.executors import DaskExecutor, LocalExecutor
from prefect.run_configs import LocalRun


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]
data_dir = Path(project_dir) / 'data'

dask_enabled = config["main"]["dask_enabled"]
prefect_cloud_enabled = config["main"]["prefect_cloud_enabled"]


# ---------------------------------------------------------


def download_file(url, path, overwrite=False):
    if not path.exists() or overwrite:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return path


@task
def get_md5_contents(url, path):
    md5_path = download_file(url, path)
    md5 = open(md5_path,'rb').read().decode("utf-8").split(' ')[0]
    return md5


@task
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

    if not valid:
        raise Exception('Failed to download and verify:', dl_url)

    return dl_path


@task
def unzip(zip_path, out_dir):
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extractall(out_dir)

    return out_dir


@task
def shapefile_to_spatialite(fname, dir, table_name):
    shp_path = dir / f'{fname}.shp'
    sqlite_path = dir / f'{fname}.sqlite'
    if not sqlite_path.exists():
        buildings_call_str = f'ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln {table_name} -dsco SPATIALITE=YES {sqlite_path} {shp_path}'

        buildings_call = sp.run(buildings_call_str, shell=True, capture_output=True)

        if buildings_call.returncode != 0:
            raise Exception(buildings_call.stderr, buildings_call_str)


# ---------------------------------------------------------


osm_date = '220101'

base_url = 'http://download.geofabrik.de'

# key is region, value is list of country names in region
country_dict = {
    'africa': ['Liberia', 'Sierra Leone', 'Cameroon', 'Guinea', 'Mali', 'Nigeria', 'Zambia', 'Benin', 'Burundi', 'Ethiopia', 'South Africa', 'Uganda', 'Angola', 'Malawi', 'Rwanda', 'Tanzania', 'Zimbabwe', 'Chad', 'Ghana', 'Kenya', 'Lesotho'],
    'antarctica': [],
    'asia': ['Jordan', 'Pakistan', 'Philippines'],
    'australia': [],
    'central-america': [],
    'europe': [],
    'north-america': [],
    'south-america': []
}

download_list = []
for region, country_names in country_dict.items():
    download_list.extend([(region, i.lower().replace(' ', '-'), base_url, data_dir) for i in country_names])


with Flow("download-osm") as flow:
    for input in download_list:
        region, country_name, base_url, data_dir = input

        fname = f'{country_name}-{osm_date}-free.shp'

        output_dir = data_dir / 'osm'

        shp_url = f'{base_url}/{region}/{fname}.zip'
        shp_path = output_dir / f'{fname}.zip'

        md5_url = f'{base_url}/{region}/{fname}.zip.md5'
        md5_path = output_dir / f'{fname}.zip.md5'

        unzip_dir = output_dir / fname


        true_md5 = get_md5_contents(md5_url, md5_path)

        shp_path = download_and_verify(shp_url, shp_path, true_md5)

        unzip_dir = unzip(shp_path, unzip_dir)

        sqlite_table_name = 'DATA_TABLE'

        osm_buildings_sqlite_fname = 'gis_osm_buildings_a_free_1'
        shapefile_to_spatialite(osm_buildings_sqlite_fname, unzip_dir, sqlite_table_name)

        osm_roads_sqlite_fname = 'gis_osm_roads_free_1'
        shapefile_to_spatialite(osm_roads_sqlite_fname, unzip_dir, sqlite_table_name)



if dask_enabled:
    executor = DaskExecutor(address="tcp://127.0.0.1:8786")
else:
    executor = LocalExecutor()

# flow.run_config = LocalRun()
flow.executor = executor

if prefect_cloud_enabled:
    flow_id = flow.register(project_name="accessible-poverty-estimates")
    client = Client()
    run_id = client.create_flow_run(flow_id=flow_id)

else:
    state = flow.run()
