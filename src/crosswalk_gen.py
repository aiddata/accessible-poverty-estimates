"""
python 3.9

Run this script to prepare the crosswalk files for each feature class in order to link original OSM types to higher level groups

This will generate new crosswalk files if one does not exist, or update an existing one with new OSM types if needed

After running this script, you will need to review the crosswalk files to assign groups for new OSM types (group = 0 by default)
    - This can be done manually using Excel or other spreadsheet software, or any other method you prefer

"""

import os
import configparser

import pandas as pd
import geopandas as gpd
import numpy as np

if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]

project_dir = config[project]["project_dir"]
dhs_round = config[project]['dhs_round']
country_name = config[project]["country_name"]
osm_date = config[project]["osm_date"]


data_dir = os.path.join(project_dir, 'data')



# ---------------------------------------------------------
# pois

print("Running pois...")

osm_pois_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_pois_free_1.shp')
osm_pois_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_pois_a_free_1.shp')

raw_pois_geo = gpd.read_file(osm_pois_shp_path)
raw_pois_a_geo = gpd.read_file(osm_pois_a_shp_path)

pois_geo = pd.concat([raw_pois_geo, raw_pois_a_geo])


pois_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/pois_type_crosswalk.csv')


type_list = list(set(pois_geo["fclass"]))

if not os.path.exists(pois_type_crosswalk_path):
    pois_type_crosswalk_df =  pd.DataFrame({"type": type_list})
    pois_type_crosswalk_df['group'] = 0

else:

    existing_pois_type_crosswalk_df = pd.read_csv(pois_type_crosswalk_path)

    new_types = [i for i in type_list if i not in existing_pois_type_crosswalk_df.type.to_list()]

    print(f'{len(new_types)} found for pois')

    # deduplicate types
    if new_types:
        pois_type_crosswalk_df = pd.concat([existing_pois_type_crosswalk_df, pd.DataFrame({"type": new_types})])
        pois_type_crosswalk_df.group.fillna(0, inplace=True)


pois_type_crosswalk_df.to_csv(pois_type_crosswalk_path, index=False)


# ---------------------------------------------------------
# traffic

print("Running traffic...")

osm_traffic_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_traffic_free_1.shp')
osm_traffic_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_traffic_a_free_1.shp')

raw_traffic_geo = gpd.read_file(osm_traffic_shp_path)
raw_traffic_a_geo = gpd.read_file(osm_traffic_a_shp_path)

traffic_geo = pd.concat([raw_traffic_geo, raw_traffic_a_geo])


traffic_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/traffic_type_crosswalk.csv')


type_list = list(set(traffic_geo["fclass"]))

if not os.path.exists(traffic_type_crosswalk_path):
    traffic_type_crosswalk_df =  pd.DataFrame({"type": type_list})
    traffic_type_crosswalk_df['group'] = 0

else:

    existing_traffic_type_crosswalk_df = pd.read_csv(traffic_type_crosswalk_path)

    new_types = [i for i in type_list if i not in existing_traffic_type_crosswalk_df.type.to_list()]

    print(f'{len(new_types)} new types found for traffic')

    # deduplicate types
    if new_types:
        traffic_type_crosswalk_df = pd.concat([existing_traffic_type_crosswalk_df, pd.DataFrame({"type": new_types})])
        traffic_type_crosswalk_df.group.fillna(0, inplace=True)


traffic_type_crosswalk_df.to_csv(traffic_type_crosswalk_path, index=False)




# ---------------------------------------------------------
# transport

print("Running transport...")

osm_transport_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_transport_free_1.shp')
osm_transport_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_transport_a_free_1.shp')

raw_transport_geo = gpd.read_file(osm_transport_shp_path)
raw_transport_a_geo = gpd.read_file(osm_transport_a_shp_path)

transport_geo = pd.concat([raw_transport_geo, raw_transport_a_geo])


transport_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/transport_type_crosswalk.csv')


type_list = list(set(transport_geo["fclass"]))

if not os.path.exists(transport_type_crosswalk_path):
    transport_type_crosswalk_df =  pd.DataFrame({"type": type_list})
    transport_type_crosswalk_df['group'] = 0

else:

    existing_transport_type_crosswalk_df = pd.read_csv(transport_type_crosswalk_path)

    new_types = [i for i in type_list if i not in existing_transport_type_crosswalk_df.type.to_list()]

    print(f'{len(new_types)} new types found for transport')

    # deduplicate types
    if new_types:
        transport_type_crosswalk_df = pd.concat([existing_transport_type_crosswalk_df, pd.DataFrame({"type": new_types})])
        transport_type_crosswalk_df.group.fillna(0, inplace=True)


transport_type_crosswalk_df.to_csv(transport_type_crosswalk_path, index=False)



# ---------------------------------------------------------
# buildings

print("Running buildings...")

osm_buildings_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.shp')
buildings_geo = gpd.read_file(osm_buildings_shp_path)


buildings_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/buildings_type_crosswalk.csv')


type_list = list(set(buildings_geo["fclass"]))

if not os.path.exists(buildings_type_crosswalk_path):
    buildings_type_crosswalk_df =  pd.DataFrame({"type": type_list})
    buildings_type_crosswalk_df['group'] = 0

else:

    existing_buildings_type_crosswalk_df = pd.read_csv(buildings_type_crosswalk_path)

    new_types = [i for i in type_list if i not in existing_buildings_type_crosswalk_df.type.to_list()]

    print(f'{len(new_types)} new types found for buildings')

    # deduplicate types
    if new_types:
        buildings_type_crosswalk_df = pd.concat([existing_buildings_type_crosswalk_df, pd.DataFrame({"type": new_types})])
        buildings_type_crosswalk_df.group.fillna(0, inplace=True)


buildings_type_crosswalk_df.to_csv(buildings_type_crosswalk_path, index=False)



# ---------------------------------------------------------
# roads

print("Running roads...")

osm_roads_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.shp')
roads_geo = gpd.read_file(osm_roads_shp_path)



roads_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/roads_type_crosswalk.csv')


type_list = list(set(roads_geo["fclass"]))

if not os.path.exists(roads_type_crosswalk_path):
    roads_type_crosswalk_df =  pd.DataFrame({"type": type_list})
    roads_type_crosswalk_df['group'] = 0

else:

    existing_roads_type_crosswalk_df = pd.read_csv(roads_type_crosswalk_path)

    new_types = [i for i in type_list if i not in existing_roads_type_crosswalk_df.type.to_list()]

    # deduplicate types
    if new_types:
        print(f'{len(new_types)} new types found for roads')
        roads_type_crosswalk_df = pd.concat([existing_roads_type_crosswalk_df, pd.DataFrame({"type": new_types})])
        roads_type_crosswalk_df.group.fillna(0, inplace=True)


roads_type_crosswalk_df.to_csv(roads_type_crosswalk_path, index=False)

