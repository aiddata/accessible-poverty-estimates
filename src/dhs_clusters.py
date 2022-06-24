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
def gen_var_dict(dhs_hh_file_name):
    # key = column, value = descriptor (used as column name)
    #   - cluster id for each DHS file used is always required (e.g., both household recode and indivdual recode if using data from both)
    #   - at least one additional column is required to represent outcome variable / indicator
    #   - source field indicates which DHS file the variable is in. No source is required for hv001
    #   - Note: this has not been tested (nor are there config vars) for use with DHS files other than the household recode
    var_dict = {
        "hv001": {
            "name": "Cluster number",
            "agg": "mean",
            "source": dhs_hh_file_name
        },
        "hv271": {
            "name": 'Wealth Index',
            "agg": "mean",
            "source": dhs_hh_file_name
        },
        # "hv108_01": {
        #     "name": "Education completed (years)",
        #     "agg": "mean",
        #     "replacements": [(996, 0), (1,3)] # 996 = water is on property
        # },
        # "hv206": {
        #     "name": "Access to electricity",
        #     "agg": "mean"
        # },
        # "hv204": {
        #     "name": "Access to water (minutes",
        #     "agg": "mean"
        # },
        # "hv208": {
        #     "name": "Has television",
        #     "agg": "mean"
        # },
        # "hv226": {
        #     "name": "Type of cooking fuel",
        #     "agg": "mean"
        # },
        # "v001": {
        #     "name": "Cluster number",
        #     "agg": "mean",
        #     "source": dhs_ir_file_name
        # },
        # "v171b": {
        #     "name": 'Internet usage in the last month',
        #     "agg": "mean",
        #     "source": dhs_ir_file_name
        # },
    }
    return var_dict


@task
def create_extract_file(output_name, data_dir):
    """create extract job file to use with resulting dhs buffers geojson for geoquery extract
    """
    iso2 = output_name.lower().split('_')[0]
    base_extract_job_path = data_dir / 'extract_job.json'
    extract_job_path = data_dir / 'outputs' / output_name / 'extract_job.json'
    extract_job_text = base_extract_job_path.read_text().replace('[[ISO2]]', iso2).replace('[[DHS_ROUND]]', output_name)
    extract_job_path.write_text(extract_job_text)


@task(nout=1)
def load_dhs_data(var_dict, data_dir):
    """
    Loads the DHS data
    """
    var_list = var_dict.keys()
    replacement_rules = [(field, rule) for field, value in var_dict.items() for rule in value.get("replacements", [])]
    agg_rules = {k: v['agg'] for k,v in var_dict.items()}
    field_names = [v['name'] for v in var_dict.values()]

    raw_dhs_list = []
    for s in set([i.get('source') for i in var_dict.values() if i.get('source')]):
        tmp_path = glob.glob(str(data_dir / 'dhs' / '**' / f'{s}.DTA'), recursive=True)[0]
        tmp_df = pd.read_stata(tmp_path, convert_categoricals=False)
        raw_dhs_list.append(tmp_df.set_index('hhid'))

    if not raw_dhs_list:
        raise Exception('No DHS data found')
    elif len(raw_dhs_list) == 1:
        raw_dhs_df = raw_dhs_list[0]
    else:
        raw_dhs_df = pd.concat(raw_dhs_list, axis=1, join='inner')

    dhs_df = raw_dhs_df[var_list].copy(deep=True)

    for field, rule in replacement_rules:
        dhs_df[field] = dhs_df[field].replace(*rule)

    cluster_data = dhs_df.groupby("hv001").agg(agg_rules).drop(columns="hv001").reset_index().dropna(axis=1)
    cluster_data.columns = field_names

    print(f'{len(dhs_df)} households aggregated to {len(cluster_data)} clusters')
    print('\tCluster data dimensions: {}'.format(cluster_data.shape))

    return cluster_data


@task
def load_dhs_geo(dhs_geo_file_name, data_dir):
    """Load shapefile containing DHS cluster points
    """
    shp_path = glob.glob(str(data_dir / 'dhs' / '**' / dhs_geo_file_name / '*.shp'), recursive=True)[0]
    gdf = gpd.read_file(shp_path)
    # drop locations without coordinates
    gdf = gdf.loc[(gdf["LONGNUM"] != 0) & (gdf["LATNUM"] != 0)].reset_index(drop=True)
    gdf.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"}, inplace=True)
    return gdf


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


@task
def buffer_dhs_clusters(gdf, country_utm_epsg_code):
    gdf = gdf.to_crs(f"EPSG:{country_utm_epsg_code}")
    gdf.geometry = gdf.apply(lambda x: buffer(x.geometry, x.URBAN_RURA), axis=1)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


@task
def merge_dhs_data(df, gdf):
    """Merge dhs data with cluster geometries

    Args:
        df (DataFrame): dhs data
        gdf (GeoDataFrame): cluster geometries

    Returns:
        GeoDataFrame: merged data
    """
    merged = gdf.merge(df, left_on="DHSCLUST", right_on="Cluster number", how="inner")
    return merged


@task
def export_data(gdf, output_name, data_dir):
    """Export data to shapefile and csv

    Args:
        gdf (GeoDataFrame): GeoDataFrame to export
        output_name (str): name of outputs directory
        data_dir (str): path to base data directory

    Returns:
        None
    """
    # output all dhs cluster data to CSV
    final_df = gdf[[i for i in gdf.columns if i != "geometry"]]
    final_path = data_dir / 'outputs' / output_name / 'dhs_data.csv'
    final_df.to_csv(final_path, index=False, encoding='utf-8')

    # save buffer geometry as geojson
    geo_path = data_dir / 'outputs' / output_name / 'dhs_buffers.geojson'
    gdf[['DHSID', 'geometry']].to_file(geo_path, driver='GeoJSON')



# ---------------------------------------------------------


if 'combination' in config[project] and config[project]['combination'] == 'True':
    dhs_list = config[project]['project_list'].replace(' ', '').split(',')
else:
    dhs_list = [project]



with Flow("dhs-clusters") as flow:

    for dhs_item in dhs_list:

        output_name = config[dhs_item]['output_name']
        dhs_hh_file_name = config[dhs_item]['dhs_hh_file_name']
        dhs_geo_file_name = config[dhs_item]['dhs_geo_file_name']
        country_utm_epsg_code = config[dhs_item]['country_utm_epsg_code']
        var_dict = gen_var_dict(dhs_hh_file_name)

        (data_dir / 'outputs' / output_name).mkdir(exist_ok=True)

        # create extract job file to use with resulting dhs buffers geojson for geoquery extract
        create_extract_file(output_name, data_dir)

        # prepare DHS cluster indicators
        cluster_data = load_dhs_data(var_dict, data_dir)

        # load DHS cluster coordinates
        raw_gdf = load_dhs_geo(dhs_geo_file_name, data_dir)

        # convert to UTM first (meters) to buffer, then back to WGS84 (degrees)
        gdf = buffer_dhs_clusters(raw_gdf, country_utm_epsg_code)

        # merge geospatial data with dhs hr indicators
        gdf_merge = merge_dhs_data(cluster_data, gdf)

        # export
        export_data(gdf_merge, output_name, data_dir)



state = run_flow(flow, executor, prefect_cloud_enabled, prefect_project_name)
