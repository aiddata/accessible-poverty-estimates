"""
python 3.9

Run this script to prepare the crosswalk files for each feature class in order to link original OSM types to higher level groups

This will generate new crosswalk files if one does not exist, or update an existing one with new OSM types if needed

After running this script, you will need to review the crosswalk files to assign groups for new OSM types (group = 0 by default)
    - This can be done manually using Excel or other spreadsheet software, or any other method you prefer

"""

import os
import configparser
from pathlib import Path

import pandas as pd
import geopandas as gpd
from dbfread import DBF


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project_dir = Path(config["main"]["project_dir"])
data_dir = project_dir / 'data'

# uncomment below to check only one country specified by the config file
# project = config["main"]["project"]
# country_name = config[project]["country_name"]
# osm_date = config[project]["osm_date"]
# dir_list = [(country_name, osm_date)]

# uncomment below to check every OSM download in the data/osm directory
dir_list = [('-'.join(i.stem.split('-')[0:-2]), i.stem.split('-')[-2]) for i in (Path(data_dir) / 'osm').glob('*.shp')]


task_list = []
for country_name, osm_date in dir_list:
    print(country_name, osm_date)
    osm_dir = data_dir / 'osm' / f'{country_name}-{osm_date}-free.shp'
    tmp_tasks = [
        (country_name, osm_date, 'pois', osm_dir / 'gis_osm_pois_free_1.dbf'),
        (country_name, osm_date, 'pois', osm_dir / 'gis_osm_pois_a_free_1.dbf'),
        (country_name, osm_date, 'traffic', osm_dir / 'gis_osm_traffic_free_1.dbf'),
        (country_name, osm_date, 'traffic', osm_dir / 'gis_osm_traffic_a_free_1.dbf'),
        (country_name, osm_date, 'transport', osm_dir / 'gis_osm_transport_free_1.dbf'),
        (country_name, osm_date, 'transport', osm_dir / 'gis_osm_transport_a_free_1.dbf'),
        (country_name, osm_date, 'buildings', osm_dir / 'gis_osm_buildings_a_free_1.dbf'),
        (country_name, osm_date, 'roads', osm_dir / 'gis_osm_roads_free_1.dbf')
    ]
    task_list.extend(tmp_tasks)



def gen_groups(country_name, osm_date, type, path):

    print(f'Running {country_name} ({osm_date}) - {type}')

    # type_table = pd.DataFrame(DBF(path, encoding='latin-1'))
    # type_list = list(set(type_table["fclass"]))
    type_list = list(set([i['fclass'] for i in DBF(path, encoding='latin-1')]))

    crosswalk_path = data_dir / f'crosswalks/{type}_type_crosswalk.csv'

    if not os.path.exists(crosswalk_path):
        type_crosswalk_df =  pd.DataFrame({"type": type_list})
        type_crosswalk_df['group'] = 0

    else:
        existing_type_crosswalk_df = pd.read_csv(crosswalk_path)
        new_types = [i for i in type_list if i not in existing_type_crosswalk_df.type.to_list()]

        # deduplicate types
        if new_types:
            print(f'\t{len(new_types)} new types found for {type}')
            type_crosswalk_df = pd.concat([existing_type_crosswalk_df, pd.DataFrame({"type": new_types})])
            type_crosswalk_df.group.fillna(0, inplace=True)
        else:
            type_crosswalk_df = existing_type_crosswalk_df

    type_crosswalk_df.to_csv(crosswalk_path, index=False)



for i in task_list:
    gen_groups(*i)
