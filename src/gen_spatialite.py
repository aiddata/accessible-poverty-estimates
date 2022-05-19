'''
Convert OSM buildgins/roads shapefiles to spatialite databases using shell calls run from python
'''


import os
import configparser
import subprocess as sp


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

data_dir = os.path.join(project_dir, 'data')

country_name = config[project]["country_name"]
osm_date = config[project]["osm_date"]



osm_buildings_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.shp')

osm_buildings_sqlite_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.sqlite')
building_table_name = 'DATA_TABLE'

buildings_call_str = f'ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln {building_table_name} -dsco SPATIALITE=YES {osm_buildings_sqlite_path} {osm_buildings_shp_path}'


buildings_call = sp.run(buildings_call_str, shell=True, capture_output=True)

if buildings_call.returncode != 0:
    raise Exception(buildings_call.stderr, buildings_call_str)



osm_roads_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.shp')

osm_roads_sqlite_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.sqlite')
road_table_name = 'DATA_TABLE'

roads_call_str = f'ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln {road_table_name} -dsco SPATIALITE=YES {osm_roads_sqlite_path} {osm_roads_shp_path}'




roads_call = sp.run(roads_call_str, shell=True, capture_output=True)

if roads_call.returncode != 0:
    raise Exception(roads_call.stderr, roads_call_str)
