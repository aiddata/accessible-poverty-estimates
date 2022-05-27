'''
Extract features for machine learning models from vector (OSM/Ecopia) data


# -------------------------------------

# convert OSM building/roads shapefiles to spatialite databases first
# all files are in 4326, utm codes in original file names just indicate utm zone splits of data


ogr2ogr -f SQLite -dsco SPATIALITE=YES africa_ghana_road.sqlite africa_ghana_road_4326/africa_ghana_road_4326.shp

# merge multiple files into one
ogr2ogr -f SQLite -dsco SPATIALITE=YES africa_ghana_building.sqlite africa_ghana_building_32630/africa_ghana_building_32630.shp
ogr2ogr -f SQLite -update -append africa_ghana_building.sqlite africa_ghana_building_32631/africa_ghana_building_32631.shp -nln merge

# to deal with multipolygon issues
ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -dsco SPATIALITE=YES africa_ghana_road.sqlite africa_ghana_road_4326/africa_ghana_road_4326.shp

# set generic table name
ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln DATA_TABLE -dsco SPATIALITE=YES africa_ghana_road.sqlite africa_ghana_road_4326/africa_ghana_road_4326.shp

# full example for osm buildings/road data
country=kenya
osm_data=220101
dir=data/osm/${country}-${osm_data}-free.shp
ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln DATA_TABLE -dsco SPATIALITE=YES ${dir}/gis_osm_buildings_a_free_1.sqlite ${dir}//gis_osm_buildings_a_free_1.shp
ogr2ogr -f SQLite -nlt PROMOTE_TO_MULTI -nln DATA_TABLE -dsco SPATIALITE=YES ${dir}/gis_osm_roads_free_1.sqlite ${dir}//gis_osm_roads_free_1.shp


# -------------------------------------

# add steps to build latest spatialite from source?
#

# spatialite info:
#   https://www.gaia-gis.it/fossil/libspatialite/index


'''

import os
import configparser
import time
import datetime
import warnings
import sqlite3

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = config["main"]["project_dir"]

output_name = config[project]['output_name']
country_utm_epsg_code = config[project]['country_utm_epsg_code']

country_name = config[project]["country_name"]
osm_date = config[project]["osm_date"]
geom_id = config[project]["geom_id"]
geom_label = config[project]["geom_label"]


data_dir = os.path.join(project_dir, 'data')

osm_features_dir = os.path.join(data_dir, 'outputs', output_name, 'osm_features')
os.makedirs(osm_features_dir, exist_ok=True)



# DHS CLUSTERS

# load buffers/geom created during data prep
geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
buffers_gdf = gpd.read_file(geom_path)

# calculate area of each buffer
# convert to UTM first, then back to WGS84 (degrees)
buffers_gdf = buffers_gdf.to_crs(epsg=country_utm_epsg_code)
buffers_gdf["buffer_area"] = buffers_gdf.area
buffers_gdf = buffers_gdf.to_crs("EPSG:4326") # WGS84
buffers_gdf['longitude'] = buffers_gdf.centroid.x
buffers_gdf['latitude'] = buffers_gdf.centroid.y
buffers_gdf['centroid_wkt'] = buffers_gdf.geometry.centroid.apply(lambda x: x.wkt)





# =============================================================================
# generic functions

def build_gpkg_connection(sqlite_path):

    # create connection to SQLite database
    conn = sqlite3.connect(sqlite_path)

    # allow SQLite to load extensions
    conn.enable_load_extension(True)

    # load SpatiaLite extension
    # see README.md for more information
    conn.load_extension(config["main"]["spatialite_lib_path"])

    # initialise spatial table support
    # conn.execute('SELECT InitSpatialMetadata(1)')

    # this statement creates missing system tables,
    # including knn2, which we will use
    conn.execute('SELECT CreateMissingSystemTables(1)')

    conn.execute('SELECT EnableGpkgAmphibiousMode()')

    return conn


def _task_wrapper(func, args):
    try:
        result = func(*args)
        return (0, "Success", args, result)
    except Exception as e:
        # raise
        return (1, repr(e), args, None)


def run_tasks(func, flist, parallel, max_workers=None, chunksize=1):
    # run all downloads (parallel and serial options)
    wrapper_list = [(func, i) for i in flist]
    if parallel:
        # https://docs.python.org/3/library/concurrent.futures.html
        from concurrent.futures import ProcessPoolExecutor
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
            warnings.warn(f"Parallel set to True (mpi4py is not installed) but max_workers not specified. Defaulting to CPU count ({max_workers})")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_gen = executor.map(_task_wrapper, *zip(*wrapper_list), chunksize=chunksize)
        results = list(results_gen)
    else:
        results = []
        # for i in flist:
            # results.append(func(*i))
        for i in wrapper_list:
            results.append(_task_wrapper(*i))
    return results


# =============================================================================
# generate features


# ---------------------------------------------------------
# pois
# count of each type of pois (100+) in each buffer

print("Running pois...")

osm_pois_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_pois_free_1.shp')
osm_pois_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_pois_a_free_1.shp')

raw_pois_geo = gpd.read_file(osm_pois_shp_path)
raw_pois_a_geo = gpd.read_file(osm_pois_a_shp_path)

pois_geo_raw = pd.concat([raw_pois_geo, raw_pois_a_geo])

# load crosswalk for types and assign any not grouped to "other"
pois_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/pois_type_crosswalk.csv')
pois_type_crosswalk_df = pd.read_csv(pois_type_crosswalk_path)
pois_type_crosswalk_df.loc[pois_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
pois_geo = pois_geo_raw.merge(pois_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

pois_geo.loc[pois_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(pois_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# pois_group_list = ["all"] + [i for i in set(pois_geo[group_field])]
pois_group_list = [i for i in set(pois_geo[group_field]) if pd.notnull(i)]

# copy of buffers gdf to use for output
buffers_gdf_pois = buffers_gdf.copy(deep=True)

for group in pois_group_list:
    print(group)
    # subet by group
    if group == "all":
        pois_geo_subset = pois_geo.reset_index(inplace=True).copy(deep=True)
    else:
        pois_geo_subset = pois_geo.loc[pois_geo[group_field] == group].reset_index().copy(deep=True)
    # query to find pois in each buffer
    bquery = pois_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
    # pois dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "pois": bquery[1]})
    # add pois data to spatial query dataframe
    bquery_full = bquery_df.merge(pois_geo_subset, left_on="pois", right_index=True, how="left")
    # aggregate spatial query df with pois info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({"pois": "count"})
    bquery_agg.columns = [group + "_pois_count"]
    # join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
    # not each cluster will have relevant pois, set those to zero
    z1.fillna(0, inplace=True)
    # set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[group + "_pois_count"]
    # merge group columns back to main cluster dataframe
    buffers_gdf_pois = buffers_gdf_pois.merge(z2, left_index=True, right_index=True)

# output final features
pois_feature_cols = [i for i in buffers_gdf_pois.columns if "_pois_" in i]
pois_cols = [geom_id] + pois_feature_cols
pois_features = buffers_gdf_pois[pois_cols].copy(deep=True)
pois_features['all_pois_count'] = pois_features[pois_feature_cols].sum(axis=1)
pois_features_path = os.path.join(osm_features_dir, '{}_pois_{}.csv'.format(geom_label, osm_date))
pois_features.to_csv(pois_features_path, index=False, encoding="utf-8")


# ---------------------------------------------------------
# traffic
# count of each type of traffic item in each buffer

print("Running traffic...")

osm_traffic_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_traffic_free_1.shp')
osm_traffic_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_traffic_a_free_1.shp')

raw_traffic_geo = gpd.read_file(osm_traffic_shp_path)
raw_traffic_a_geo = gpd.read_file(osm_traffic_a_shp_path)

traffic_geo_raw = pd.concat([raw_traffic_geo, raw_traffic_a_geo])

# load crosswalk for types and assign any not grouped to "other"
traffic_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/traffic_type_crosswalk.csv')
traffic_type_crosswalk_df = pd.read_csv(traffic_type_crosswalk_path)
traffic_type_crosswalk_df.loc[traffic_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
traffic_geo = traffic_geo_raw.merge(traffic_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

traffic_geo.loc[traffic_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(traffic_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# traffic_group_list = ["all"] + [i for i in set(traffic_geo[group_field])]
traffic_group_list = [i for i in set(traffic_geo[group_field]) if pd.notnull(i)]

# copy of buffers gdf to use for output
buffers_gdf_traffic = buffers_gdf.copy(deep=True)

for group in traffic_group_list:
    print(group)
    # subet by group
    if group == "all":
        traffic_geo_subset = traffic_geo.copy(deep=True)
    else:
        traffic_geo_subset = traffic_geo.loc[traffic_geo[group_field] == group].reset_index().copy(deep=True)
    # query to find traffic in each buffer
    bquery = traffic_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
    # traffic dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "traffic": bquery[1]})
    # add traffic data to spatial query dataframe
    bquery_full = bquery_df.merge(traffic_geo_subset, left_on="traffic", right_index=True, how="left")
    # aggregate spatial query df with traffic info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({"traffic": "count"})
    bquery_agg.columns = [group + "_traffic_count"]
    # join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
    # not each cluster will have relevant traffic, set those to zero
    z1.fillna(0, inplace=True)
    # set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[group + "_traffic_count"]
    # merge group columns back to main cluster dataframe
    buffers_gdf_traffic = buffers_gdf_traffic.merge(z2, left_index=True, right_index=True)

# output final features
traffic_feature_cols = [i for i in buffers_gdf_traffic.columns if "_traffic_" in i]
traffic_cols = [geom_id] + traffic_feature_cols
traffic_features = buffers_gdf_traffic[traffic_cols].copy(deep=True)
traffic_features['all_traffic_count'] = traffic_features[traffic_feature_cols].sum(axis=1)
traffic_features_path = os.path.join(osm_features_dir, '{}_traffic_{}.csv'.format(geom_label, osm_date))
traffic_features.to_csv(traffic_features_path, index=False, encoding="utf-8")

# ---------------------------------------------------------
# transport
# count of each type of transport item in each buffer

print("Running transport...")

osm_transport_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_transport_free_1.shp')
osm_transport_a_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_transport_a_free_1.shp')

raw_transport_geo = gpd.read_file(osm_transport_shp_path)
raw_transport_a_geo = gpd.read_file(osm_transport_a_shp_path)

transport_geo_raw = pd.concat([raw_transport_geo, raw_transport_a_geo])

# load crosswalk for types and assign any not grouped to "other"
transport_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/transport_type_crosswalk.csv')
transport_type_crosswalk_df = pd.read_csv(transport_type_crosswalk_path)
transport_type_crosswalk_df.loc[transport_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
transport_geo = transport_geo_raw.merge(transport_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

transport_geo.loc[transport_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(transport_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# transport_group_list = ["all"] + [i for i in set(transport_geo[group_field])]
transport_group_list = [i for i in set(transport_geo[group_field]) if pd.notnull(i)]

# copy of buffers gdf to use for output
buffers_gdf_transport = buffers_gdf.copy(deep=True)

for group in transport_group_list:
    print(group)
    # subet by group
    if group == "all":
        transport_geo_subset = transport_geo.copy(deep=True)
    else:
        transport_geo_subset = transport_geo.loc[transport_geo[group_field] == group].reset_index().copy(deep=True)
    # query to find transport in each buffer
    bquery = transport_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
    # transport dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "transport": bquery[1]})
    # add transport data to spatial query dataframe
    bquery_full = bquery_df.merge(transport_geo_subset, left_on="transport", right_index=True, how="left")
    # aggregate spatial query df with transport info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({"transport": "count"})
    bquery_agg.columns = [group + "_transport_count"]
    # join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
    # not each cluster will have relevant transport, set those to zero
    z1.fillna(0, inplace=True)
    # set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[group + "_transport_count"]
    # merge group columns back to main cluster dataframe
    buffers_gdf_transport = buffers_gdf_transport.merge(z2, left_index=True, right_index=True)

# output final features
transport_feature_cols = [i for i in buffers_gdf_transport.columns if "_transport_" in i]
transport_cols = [geom_id] + transport_feature_cols
transport_features = buffers_gdf_transport[transport_cols].copy(deep=True)
transport_features['all_transport_count'] = transport_features[transport_feature_cols].sum(axis=1)
transport_features_path = os.path.join(osm_features_dir, '{}_transport_{}.csv'.format(geom_label, osm_date))
transport_features.to_csv(transport_features_path, index=False, encoding="utf-8")


# ---------------------------------------------------------
# buildings

buffers_gdf_buildings = buffers_gdf.copy(deep=True)

building_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.sqlite')
building_table_name = 'DATA_TABLE'


sqlite_building_conn = build_gpkg_connection(building_path)
# sqlite_building_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
# sqlite_building_conn.execute(f'PRAGMA table_info({building_table_name})').fetchall()


# load crosswalk for building types and assign any not grouped to "other"
building_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/buildings_type_crosswalk.csv')
building_type_crosswalk_df = pd.read_csv(building_type_crosswalk_path)
building_type_crosswalk_df.loc[building_type_crosswalk_df["group"] == "0", "group"] = "other"
building_type_crosswalk_df = building_type_crosswalk_df.loc[building_type_crosswalk_df.type.notna()]

building_group_lists = building_type_crosswalk_df.groupby('group').agg({'type':list}).reset_index()

for i in building_group_lists.itertuples():
    _, group, type_list = i
    q = f'''
        SELECT ogc_fid
        FROM {building_table_name}
        WHERE type in {tuple(type_list)}
        '''
    r = pd.read_sql(q, sqlite_building_conn)
    print(group, len(r))




# # geopandas approach
# osm_buildings_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.shp')
# buildings_geo_raw = gpd.read_file(osm_buildings_shp_path)
# group_field = "group"

# # merge new classification and assign any buildings without a type to unclassifid
# buildings_geo = buildings_geo_raw.merge(building_type_crosswalk_df, on="type", how="left")

# buildings_group_list = [i for i in set(buildings_geo[group_field]) if i not in ["other", "unclassified"]]

# if "all" not in buildings_group_list:
#     buildings_geo = buildings_geo.loc[buildings_geo[group_field].isin(buildings_group_list)]

# # calculate area of each building
# # convert to UTM first, then back to WGS84 (degrees)
# buildings_geo = buildings_geo.to_crs(f"EPSG:{country_utm_epsg_code}")
# buildings_geo["area"] = buildings_geo.area
# buildings_geo = buildings_geo.to_crs("EPSG:4326") # WGS84

# # copy of buffers gdf to use for output
# buffers_gdf_buildings = buffers_gdf.copy(deep=True)

# for group in buildings_group_list:
#     print(group)
# for i in building_group_lists.itertuples():
#     _, group, type_list = i
#     print(f'Buildings ({group})')
#     # subet by group
#     if group == "all":
#         buildings_geo_subset = buildings_geo.copy(deep=True)
#     else:
#         buildings_geo_subset = buildings_geo.loc[buildings_geo[group_field] == group].reset_index().copy(deep=True)
#     # query to find buildings in each buffer
#     bquery = buildings_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
#     # building dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
#     bquery_df = pd.DataFrame({"cluster": bquery[0], "building": bquery[1]})
#     # add building data to spatial query dataframe
#     bquery_full = bquery_df.merge(buildings_geo_subset, left_on="building", right_index=True, how="left")
#     # aggregate spatial query df with building info, by cluster
#     bquery_agg = bquery_full.groupby("cluster").agg({
#         "area": ["count", "mean", "sum"]
#     })
#     # rename agg df
#     basic_building_cols = ["buildings_count", "buildings_avgarea", "buildings_totalarea"]
#     bquery_agg.columns = ["{}_{}".format(group, i) for i in basic_building_cols]
#     # join cluster back to original buffer_geo dataframe with columns for specific building type queries
#     z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
#     # not each cluster will have relevant buildings, set those to zero
#     z1.fillna(0, inplace=True)
#     # calculate ratio for building type
#     z1["{}_buildings_ratio".format(group)] = z1["{}_buildings_totalarea".format(group)] / z1["buffer_area"]
#     # set index and drop unnecessary columns
#     if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
#     z2 = z1[bquery_agg.columns.to_list() + ["{}_buildings_ratio".format(group)]]
#     # merge group columns back to main cluster dataframe
#     buffers_gdf_buildings = buffers_gdf_buildings.merge(z2, left_index=True, right_index=True)




# spatialite approach
for i in building_group_lists.itertuples():
    _, group, type_list = i
    print(f'Buildings ({group})')
    buffers_gdf_buildings[f"{group}_buildings_count"] = None
    buffers_gdf_buildings[f"{group}_buildings_totalarea"] = None
    buffers_gdf_buildings[f"{group}_buildings_avgarea"] = None
    buffers_gdf_buildings[f"{group}_buildings_ratio"] = None

    task_list = []

    for _, row in buffers_gdf_buildings.iterrows():

        int_wkt = row['geometry'].wkt

        q = f'''
            SELECT ogc_fid, AsText(st_intersection(geometry, GeomFromText("{int_wkt}"))) AS geometry
            FROM {building_table_name}
            WHERE type in {tuple(type_list)} AND st_intersects(geometry, GeomFromText("{int_wkt}"))
            '''

        task = (row[geom_id], q)
        task_list.append(task)


    def run_query(id, q):
        return id, pd.read_sql(q, sqlite_building_conn)


    ts = time.time()

    results_list = run_tasks(run_query, task_list, parallel=True, max_workers=12, chunksize=1)

    te = time.time()
    run_time = int(te - ts)
    print('\tquery runtime:', str(datetime.timedelta(seconds=run_time)))

    for _, _, _, results in results_list:
        row_id, result_df = results
        df = result_df.copy(deep=True)
        df['geometry'] = gpd.GeoSeries.from_wkt(df.geometry)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(epsg=country_utm_epsg_code)

        # gdf['length'] = gdf.length
        # total_length = gdf.length.sum()
        buffer_area = buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row_id, 'buffer_area'].values[0]

        buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row_id, f"{group}_buildings_count"] = gdf.shape[0]
        buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row_id, f"{group}_buildings_totalarea"] = gdf.area.sum()
        buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row_id, f"{group}_buildings_avgarea"] = gdf.area.mean()
        buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row_id, f"{group}_buildings_ratio"] = gdf.area.sum() / buffer_area



# use group results to generate "all" results
group = 'all'
buffers_gdf_buildings[f"{group}_buildings_count"] = buffers_gdf_buildings[[f"{g}_buildings_count" for g  in building_group_lists['group']]].sum(axis=1)
buffers_gdf_buildings[f"{group}_buildings_totalarea"] = buffers_gdf_buildings[[f"{g}_buildings_totalarea" for g  in building_group_lists['group']]].sum(axis=1)
buffers_gdf_buildings[f"{group}_buildings_avgarea"] = buffers_gdf_buildings[f"{group}_buildings_totalarea"] / buffers_gdf_buildings[f"{group}_buildings_count"]
buffers_gdf_buildings[f"{group}_buildings_ratio"] = buffers_gdf_buildings[f"{group}_buildings_totalarea"] / buffers_gdf_buildings['buffer_area']




buildings_cols = [geom_id] + [i for i in buffers_gdf_buildings.columns if '_buildings_' in i]
buildings_features = buffers_gdf_buildings[buildings_cols].copy(deep=True)
buildings_features.fillna(0, inplace=True)


buildings_features_path = os.path.join(osm_features_dir, f'{geom_label}_buildings_{osm_date}.csv')
buildings_features.to_csv(buildings_features_path, index=False, encoding="utf-8")


sqlite_building_conn.close()





# ---------------------------------------------------------
# road metrics


road_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.sqlite')

road_table_name = 'DATA_TABLE'

sqlite_road_conn = build_gpkg_connection(road_path)
# sqlite_road_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
# sqlite_road_conn.execute(f'PRAGMA table_info({road_table_name})').fetchall()


# load crosswalk for types and assign any not grouped to "other"
roads_type_crosswalk_path = os.path.join(data_dir, 'crosswalks/roads_type_crosswalk.csv')
roads_type_crosswalk_df = pd.read_csv(roads_type_crosswalk_path)
roads_type_crosswalk_df.loc[roads_type_crosswalk_df["group"] == "0", "group"] = "other"
roads_type_crosswalk_df = roads_type_crosswalk_df.loc[roads_type_crosswalk_df.type.notna()]

road_group_lists = roads_type_crosswalk_df.groupby('group').agg({'type':list}).reset_index()

for i in road_group_lists.itertuples():
    _, group, type_list = i
    if len(type_list) == 1:
        type_list.append('unused_random_string')
    q = f'''
        SELECT ogc_fid
        FROM {road_table_name}
        WHERE fclass in {tuple(type_list)}
        '''
    r = pd.read_sql(q, sqlite_road_conn)
    print(group, len(r))




# -------------------------------------
# distance to nearest road

# ========
# spatialite approach for nearest road
# ========

# def gen_knn_query_string(id, wkt, type_list, table_name):
#     q = f'''SELECT d.osm_id AS osm_id, d.fclass AS fclass, k.distance_m AS dist_m, k.distance_crs AS dist_crs
#     FROM KNN2 AS k
#     JOIN {table_name} AS d ON (d.ogc_fid = k.fid)
#     WHERE d.fclass IN {tuple(type_list)} AND f_table_name = '{table_name}' AND ref_geometry = PointFromText("{wkt}") AND radius = 0.5 AND max_items = 1024
#     '''
#     return [id, q]


# def find_nearest(id, q):
#     results = pd.read_sql(q, sqlite_road_conn)
#     if len(results) == 0:
#         return id, None, None
#     else:
#         return id, results.iloc[0]['osm_id'], results.iloc[0]['dist_m']


# nearest_roads_gdf_spatialite = buffers_gdf.copy(deep=True)

# for i in road_group_lists.itertuples():
#     _, group, type_list = i
#     print(f'Roads nearest({group})')

#     knn_task_list = [gen_knn_query_string(i[geom_id], i.centroid_wkt, type_list, road_table_name) for _,i in buffers_gdf.iterrows()]

#     ts = time.time()

#     results = run_tasks(find_nearest, knn_task_list, parallel=True, max_workers=12, chunksize=1)

#     te = time.time()
#     run_time = int(te - ts)
#     print('\tnearest query runtime:', str(datetime.timedelta(seconds=run_time)))

#     nearest_road_df = pd.DataFrame([i[3] for i in results], columns=[geom_id, f"{group}_roads_nearestid", f"{group}_roads_nearestdist"])
#     nearest_roads_gdf_spatialite = nearest_roads_gdf_spatialite.merge(nearest_road_df, on=geom_id, how='left')


# group = 'all'
# nearest_roads_gdf_spatialite[f"{group}_roads_nearestdist"] = nearest_roads_gdf_spatialite[[f"{g}_roads_nearestdist" for g  in road_group_lists['group']]].min(axis=1)
# nearest_roads_gdf_spatialite[f"{group}_roads_nearest-osmid"] = nearest_roads_gdf_spatialite[[f"{g}_roads_nearestdist" for g  in road_group_lists['group']]].idxmin(axis=1)





# ========
# geopandas approach for nearest road
# ========

osm_roads_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.shp')
roads_raw_geo = gpd.read_file(osm_roads_shp_path)

# merge new classification and assign any features without a type to unclassifid
roads_geo = roads_raw_geo.merge(roads_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

roads_geo.loc[roads_geo["fclass"].isna(), "group"] = "unknown"


nearest_roads_gdf_pandas = buffers_gdf.copy(deep=True)

src_points = nearest_roads_gdf_pandas.apply(lambda x: (x.longitude, x.latitude), axis=1).to_list()

group_field = 'group'

for group in road_group_lists['group']:
    print(f'Roads nearest ({group})')
    # subset based on group
    if group == "all":
        subset_roads_geo = roads_geo.copy(deep=True)
    else:
        subset_roads_geo = roads_geo.loc[roads_geo[group_field] == group].reset_index().copy(deep=True)
    # generate list of all road vertices and convert to geodataframe
    line_xy = subset_roads_geo.apply(lambda x: (x.osm_id, x.geometry.xy), axis=1)
    line_xy_lookup = [j for i in line_xy for j in list(zip([i[0]]*len(i[1][0]), *i[1]))]
    line_xy_df = pd.DataFrame(line_xy_lookup, columns=["osm_id", "x", "y"])
    line_xy_points = [(i[1], i[2]) for i in line_xy_lookup]
    # create ball tree for nearest point lookup
    #  see https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    tree = BallTree(line_xy_points, leaf_size=50, metric='haversine')
    # query tree
    distances, indices = tree.query(src_points, k=1)
    distances = distances.transpose()
    indices = indices.transpose()
    # k=1 so output length is array of len=1
    closest = indices[0]
    closest_dist = distances[0]
    # def func to get osm id for closest locations
    osm_id_lookup = lambda idx: line_xy_df.loc[idx].osm_id
    # set final data
    nearest_roads_gdf_pandas["{}_roads_nearestid".format(group)] = list(map(osm_id_lookup, closest))
    nearest_roads_gdf_pandas["{}_roads_nearestdist".format(group)] = closest_dist



nearest_roads_gdf_pandas = nearest_roads_gdf_pandas[[geom_id] + [i for i in nearest_roads_gdf_pandas.columns if "_roads_" in i]]
nearest_roads_gdf_pandas.set_index(geom_id, inplace=True)


group = 'all'

nearest_roads_gdf_pandas[f"{group}_roads_nearestdist"] = nearest_roads_gdf_pandas[[f"{g}_roads_nearestdist" for g  in road_group_lists['group']]].min(axis=1)

nearest_roads_gdf_pandas[f"{group}_roads_nearest-osmid"] = nearest_roads_gdf_pandas[[f"{g}_roads_nearestdist" for g  in road_group_lists['group']]].idxmin(axis=1)



# -------------------------------------
# length of roads in buffer


length_roads_gdf = buffers_gdf.copy(deep=True)

# # geopandas approach
# for i in road_group_lists.itertuples():
#     _, group, type_list = i
#     print(f'Roads length ({group})')
#     if group == "all":
#         subset_roads_geo = roads_geo.copy(deep=True)
#     else:
#         subset_roads_geo = roads_geo.loc[roads_geo[group_field] == group].reset_index().copy(deep=True)
#     # query to find roads in each buffer
#     bquery = subset_roads_geo.sindex.query_bulk(buffers_gdf.geometry)
#     # roads dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
#     bquery_df = pd.DataFrame({"cluster": bquery[0], "roads": bquery[1]})
#     # add roads data to spatial query dataframe
#     bquery_full = bquery_df.merge(roads_geo, left_on="roads", right_index=True, how="left")
#     # aggregate spatial query df with roads info, by cluster
#     bquery_agg = bquery_full.groupby("cluster").agg({"road_length": ["count", "sum"]})
#     bquery_agg.columns = [group + "_roads_count", group + "_roads_length"]
#     # join cluster back to original buffer_geo dataframe with columns for specific building type queries
#     z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
#     # not each cluster will have relevant roads, set those to zero
#     z1.fillna(0, inplace=True)
#     # set index and drop unnecessary columns
#     if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
#     z2 = z1[[group + "_roads_count", group + "_roads_length"]]
#     # merge group columns back to main cluster dataframe
#     length_roads_gdf = length_roads_gdf.merge(z2, left_index=True, right_index=True)


# spatialite approach
for i in road_group_lists.itertuples():
    _, group, type_list = i
    print(f'Roads length ({group})')
    length_roads_gdf[f"{group}_roads_length"] = None

    task_list = []

    for _, row in length_roads_gdf.iterrows():

        int_wkt = row['geometry'].wkt

        q = f'''
            SELECT ogc_fid, AsText(st_intersection(geometry, GeomFromText("{int_wkt}"))) AS geometry
            FROM {road_table_name}
            WHERE fclass in {tuple(type_list)} AND st_intersects(geometry, GeomFromText("{int_wkt}"))
            '''

        task = (row[geom_id], q)
        task_list.append(task)


    def run_query(id, q):
        return id, pd.read_sql(q, sqlite_road_conn)


    ts = time.time()

    results_list = run_tasks(run_query, task_list, parallel=True, max_workers=12, chunksize=1)

    te = time.time()
    run_time = int(te - ts)
    print('\tquery runtime:', str(datetime.timedelta(seconds=run_time)))

    for _, _, _, results in results_list:
        row_id, result_df = results
        df = result_df.copy(deep=True)
        df['geometry'] = gpd.GeoSeries.from_wkt(df.geometry)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(epsg=country_utm_epsg_code)

        gdf['length'] = gdf.length
        total_length = gdf.length.sum()

        length_roads_gdf.loc[length_roads_gdf[geom_id] == row_id, f"{group}_roads_length"] = total_length




# use group results to generate "all" results
group = 'all'
length_roads_gdf[f"{group}_roads_length"] = length_roads_gdf[[f"{g}_roads_length" for g  in road_group_lists['group']]].sum(axis=1)

length_roads_gdf.set_index(geom_id, inplace=True)

length_roads_gdf = length_roads_gdf[[i for i in length_roads_gdf.columns if "_roads_" in i]]




# -------------------------------------
# merge and output all roads data

nearest_roads_gdf = nearest_roads_gdf_pandas
# nearest_roads_gdf = nearest_roads_gdf_spatialite

roads_features = pd.merge(nearest_roads_gdf, length_roads_gdf, how='inner', left_index=True, right_index=True)

assert(len(roads_features) == len(buffers_gdf))


roads_features.fillna(0, inplace=True)
roads_features[geom_id] = roads_features.index

roads_features_path = os.path.join(osm_features_dir, f'{geom_label}_roads_{osm_date}.csv')
roads_features.to_csv(roads_features_path, index=False, encoding="utf-8")


sqlite_road_conn.close()


