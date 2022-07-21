import os
import configparser
import glob
from pathlib import Path
import requests

import geopandas as gpd

import prefect
from prefect import task, Flow, unmapped
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor


from utils import run_flow


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]
data_dir = Path(project_dir, 'data')

prefect_cloud_enabled = config.getboolean("main", "prefect_cloud_enabled")
prefect_project_name = config["main"]["prefect_project_name"]

dask_enabled = config.getboolean("main", "dask_enabled")
dask_distributed = config.getboolean("main", "dask_distributed") if "dask_distributed" in config["main"] else False

if dask_enabled:
    if dask_distributed:
        dask_address = config["main"]["dask_address"]
        executor = DaskExecutor(address=dask_address)
    else:
        executor = LocalDaskExecutor(scheduler="processes")
else:
    executor = LocalExecutor()


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
    adm_path = data_dir / "boundaries" / f"{iso3}_ADM{adm}_simplified.geojson"
    # create the boundaries directory, if it does not yet exist
    adm_path.parent.mkdir(exist_ok = True)
    if not adm_path.exists():

        # decided to skip actual api call and just use direct github url
        # api is available at: f'https://www.geoboundaries.org/api/v4/gbOpen/{iso3/ADM{adm}'
        # and simplified geojson url is in the 'simplifiedGeometryGeoJSON' field
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
def join_and_export(dhs_geo_file_name, dhs_gdf, adm1_gdf, adm2_gdf, data_dir, buffer=0.01):
    """Join DHS cluster points to ADM1 and ADM2 boundaries

    Buffer slightly to account for lack of overlap with some simplified/or slightly misdefined geometries
    """
    logger = prefect.context.get("logger")
    logger.info(f'{dhs_geo_file_name}')

    if buffer:
        dhs_gdf['geometry'] = dhs_gdf['geometry'].buffer(buffer)

    adm1_join = gpd.sjoin(dhs_gdf, adm1_gdf, how='left').rename(columns={'shapeID': 'ADM1'}).drop(columns=['index_right'])
    adm2_join = gpd.sjoin(adm1_join, adm2_gdf, how='left').rename(columns={'shapeID': 'ADM2'}).drop(columns=['index_right', 'geometry'])

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


state = run_flow(flow, executor, prefect_cloud_enabled, prefect_project_name)
