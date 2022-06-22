


import os
import configparser
import glob
from pathlib import Path
import requests

import geopandas as gpd

import prefect
from prefect import task, Flow, Client
from prefect.executors import DaskExecutor, LocalExecutor
from prefect.run_configs import LocalRun


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])
data_dir = project_dir / 'data'

dask_enabled = config["main"]["dask_enabled"]
prefect_cloud_enabled = config["main"]["prefect_cloud_enabled"]


# ---------------------------------------------------------

@task
def load_dhs_geo(dhs_geo_file_name, data_dir):
    """Load shapefile containing DHS cluster points
    """
    shp_path = glob.glob(str(data_dir / 'dhs' / '**' / dhs_geo_file_name / '*.shp'), recursive=True)[0]
    gdf = gpd.read_file(shp_path)
    # drop locations without coordinates
    gdf = gdf.loc[(gdf["LONGNUM"] != 0) & (gdf["LATNUM"] != 0)].reset_index(drop=True)
    gdf.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"}, inplace=True)
    return gdf[['DHSID', 'geometry']]


@task
def download_and_load_geoboundaries(iso3, adm, data_dir):
    adm_path = data_dir / 'boundaries' / f"{iso3}_ADM{adm}_simplified.geojson"
    if not adm_path.exists():

        # decided to skip actual api call and just use direct github url
        #  api is available at: f'https://www.geoboundaries.org/api/v4/gbOpen/{iso3/ADM{adm}'
        #  and simplified geojson url is in the 'simplifiedGeometryGeoJSON' field
        adm_url = f'https://github.com/wmgeolab/geoBoundaries/blob/v4.0.0/releaseData/gbOpen/{iso3}/ADM{adm}/geoBoundaries-{iso3}-ADM{adm}_simplified.geojson?raw=true'

        # stream to local output
        with requests.get(adm_url, stream=True) as r:
            r.raise_for_status()
            with open(adm_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    adm_gdf = gpd.read_file(adm_path)[['shapeID', 'geometry']]
    return adm_gdf


@task(log_stdout=True)
def join_and_export(dhs_geo_file_name, dhs_gdf, adm1_gdf, adm2_gdf, data_dir):
    logger = prefect.context.get("logger")
    logger.info(f'{dhs_geo_file_name}')

    adm1_join = dhs_gdf.sjoin(adm1_gdf, how='left', predicate='intersects').rename(columns={'shapeID': 'ADM1'}).drop(columns=['index_right'])
    adm2_join = adm1_join.sjoin(adm2_gdf, how='left', predicate='intersects').rename(columns={'shapeID': 'ADM2'}).drop(columns=['index_right', 'geometry'])

    adm2_join.to_csv(str(data_dir / 'dhs' / f'{dhs_geo_file_name}_adm_units.csv'), index=False)


# ---------------------------------------------------------



if 'combination' in config[project] and config[project]['combination'] == 'True':
    dhs_list = config[project]['project_list'].replace(' ', '').split(',')
else:
    dhs_list = [project]


with Flow("dhs-adm-units") as flow:

    for dhs_item in dhs_list:
        output_name = config[dhs_item]['output_name']
        dhs_geo_file_name = config[dhs_item]['dhs_geo_file_name']
        gb_iso3 = config[dhs_item]['gb_iso3']
        print(output_name)

        dhs_gdf = load_dhs_geo(dhs_geo_file_name, data_dir)
        adm1_gdf = download_and_load_geoboundaries(gb_iso3, 1, data_dir)
        adm2_gdf = download_and_load_geoboundaries(gb_iso3, 2, data_dir)

        join_and_export(dhs_geo_file_name, dhs_gdf, adm1_gdf, adm2_gdf, data_dir)


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

# flow.run_config = LocalRun()
# state = flow.run()
