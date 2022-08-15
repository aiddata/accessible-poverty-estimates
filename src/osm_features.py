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

from prefect import Flow, unmapped
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor

from utils import run_flow
import osm_features_tasks as oft


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]
data_dir = os.path.join(project_dir, 'data')

prefect_cloud_enabled = config["main"]["prefect_cloud_enabled"]
prefect_project_name = config["main"]["prefect_project_name"]

dask_enabled = config["main"]["dask_enabled"]
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



output_name = config[project]['output_name']
country_utm_epsg_code = config[project]['country_utm_epsg_code']

country_name = config[project]["country_name"]
osm_date = config[project]["osm_date"]
geom_id = config[project]["geom_id"]
geom_label = config[project]["geom_label"]

group_field = "group"

osm_features_dir = os.path.join(data_dir, 'outputs', output_name, 'osm_features')
os.makedirs(osm_features_dir, exist_ok=True)



with Flow("osm-features-pois") as flow:

    geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
    buffers_gdf = oft.load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    # ---------------------------------------------------------
    # pois
    # count of each type of pois (100+) in each buffer

    pois_geo_raw = oft.load_osm_shp("pois", data_dir, country_name, osm_date)

    pois_type_crosswalk_df = oft.load_crosswalk("pois", data_dir)

    pois_geo = oft.merge_crosswalk(pois_geo_raw, pois_type_crosswalk_df)

    pois_group_list = oft.get_groups(pois_geo, group_field)

    pois_group_data = oft.point_query.map(pois_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(pois_geo), osm_type=unmapped("pois"))

    buffers_gdf_pois = oft.merge_features_data(buffers_gdf, pois_group_data)

    pois_features_path = os.path.join(osm_features_dir, '{}_pois_{}.csv'.format(geom_label, osm_date))

    oft.export_point_features(buffers_gdf_pois, geom_id, 'pois', pois_features_path)


    # ---------------------------------------------------------
    # traffic
    # count of each type of traffic item in each buffer

    traffic_geo_raw = oft.load_osm_shp("traffic", data_dir, country_name, osm_date)

    traffic_type_crosswalk_df = oft.load_crosswalk("traffic", data_dir)

    traffic_geo = oft.merge_crosswalk(traffic_geo_raw, traffic_type_crosswalk_df)

    traffic_group_list = oft.get_groups(traffic_geo, group_field)

    traffic_group_data = oft.point_query.map(traffic_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(traffic_geo), osm_type=unmapped("traffic"))

    buffers_gdf_traffic = oft.merge_features_data(buffers_gdf, traffic_group_data)

    traffic_features_path = os.path.join(osm_features_dir, '{}_traffic_{}.csv'.format(geom_label, osm_date))

    oft.export_point_features(buffers_gdf_traffic, geom_id, 'traffic', traffic_features_path)


    # ---------------------------------------------------------
    # transport
    # count of each type of transport item in each buffer

    transport_geo_raw = oft.load_osm_shp("transport", data_dir, country_name, osm_date)

    transport_type_crosswalk_df = oft.load_crosswalk("transport", data_dir)

    transport_geo = oft.merge_crosswalk(transport_geo_raw, transport_type_crosswalk_df)

    transport_group_list = oft.get_groups(transport_geo, group_field)

    transport_group_data = oft.point_query.map(transport_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(transport_geo), osm_type=unmapped("transport"))

    buffers_gdf_transport = oft.merge_features_data(buffers_gdf, transport_group_data)

    transport_features_path = os.path.join(osm_features_dir, '{}_transport_{}.csv'.format(geom_label, osm_date))

    oft.export_point_features(buffers_gdf_transport, geom_id, 'transport', transport_features_path)




with Flow("osm-features-buildings") as flow:

    sqlite_building_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.sqlite')
    sqlite_lib_path = config["main"]["spatialite_lib_path"]

    sqlite_access = oft.init_spatialite(sqlite_building_path, sqlite_lib_path)

    sqlite_building_table_name = 'DATA_TABLE'
    osm_type_field = 'type'
    # sqlite_building_conn = build_gpkg_connection(sqlite_building_path)
    # sqlite_building_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
    # sqlite_building_conn.execute(f'PRAGMA table_info({sqlite_building_table_name})').fetchall()

    building_type_crosswalk_df = oft.load_crosswalk("buildings", data_dir)

    building_group_lists = oft.get_spatialite_groups(building_type_crosswalk_df, group_field, sqlite_access, sqlite_building_table_name, osm_type_field)

    geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
    buffers_gdf = oft.load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    task_list = oft.create_sqlite_task_list(building_group_lists, buffers_gdf, sqlite_building_table_name, osm_type_field)

    # results_list = [run_sqlite_query(i, sqlite_access) for i in task_list]
    results_list = oft.run_sqlite_query.map(task_list[:30], sqlite_access=unmapped(sqlite_access))

    buffers_gdf_buildings = oft.process_sqlite_results(results_list, buffers_gdf, country_utm_epsg_code, geom_id, 'buildings')

    buffers_gdf_buildings = oft.create_aggegate_metrics(buffers_gdf_buildings, building_group_lists, 'buildings')

    buildings_features_path = os.path.join(osm_features_dir, f'{geom_label}_buildings_{osm_date}.csv')

    buildings_features = oft.export_sqlite(buffers_gdf_buildings, buildings_features_path, 'buildings')




with Flow("osm-features-roads") as flow:

    sqlite_road_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.sqlite')
    sqlite_lib_path = config["main"]["spatialite_lib_path"]

    sqlite_road_table_name = 'DATA_TABLE'
    osm_type_field = 'fclass'

    road_type_crosswalk_df = oft.load_crosswalk("roads", data_dir)

    road_group_lists = oft.get_spatialite_groups(road_type_crosswalk_df, group_field, sqlite_access, sqlite_road_table_name, osm_type_field)

    geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
    buffers_gdf = oft.load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    # ---------
    #length specific
    task_list = oft.create_sqlite_task_list(road_group_lists, buffers_gdf, sqlite_road_table_name, osm_type_field)

    # results_list = [run_sqlite_query(i, sqlite_access) for i in task_list]
    results_list = oft.run_sqlite_query.map(task_list, sqlite_access=unmapped(sqlite_access))

    roads_length_gdf = oft.process_sqlite_results(results_list, buffers_gdf, country_utm_epsg_code, geom_id, 'roads')

    roads_length_gdf = oft.create_aggegate_metrics(roads_length_gdf, road_group_lists, 'roads')

    oft.flow_print(roads_length_gdf)

    # ---------
    # nearest specific

    roads_geo_raw = oft.load_osm_shp("roads", data_dir, country_name, osm_date)

    roads_geo = oft.merge_crosswalk(roads_geo_raw, road_type_crosswalk_df)

    road_group_list = oft.simple(road_group_lists)

    # nearest_results_list = [find_nearest(i, group_field, roads_geo, buffers_gdf, geom_id) for i in road_group_list]
    nearest_results_list = oft.find_nearest.map(road_group_list, group_field=unmapped(group_field), osm_gdf=unmapped(roads_geo), query_gdf=unmapped(buffers_gdf), geom_id=unmapped(geom_id))

    roads_nearest_gdf = oft.merge_road_nearest_features_data(buffers_gdf, nearest_results_list)

    # # create nearest aggregates
    roads_nearest_gdf = oft.create_aggegate_metrics(roads_nearest_gdf, road_group_lists, 'nearest')

    oft.flow_print(roads_nearest_gdf)

    # ---------


    # merge length gdf and nearest gdf
    roads_merge_gdf = oft.merge_road_features(roads_length_gdf, roads_nearest_gdf)

    roads_features_path = os.path.join(osm_features_dir, '{}_roads_{}.csv'.format(geom_label, osm_date))
    oft.export_road_features(roads_merge_gdf, geom_id, 'roads', roads_features_path)




state = run_flow(flow, executor, prefect_cloud_enabled, prefect_project_name)


















# ---------------------------------------------------------
# geopandas approach for buildings
# ---------------------------------------------------------

# osm_buildings_shp_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.shp')
# buildings_geo_raw = gpd.read_file(osm_buildings_shp_path)
# group_field = "group"

# # merge new classification and assign any buildings without a type to unclassifid
# buildings_geo = buildings_geo_raw.merge(building_type_crosswalk_df, on="type", how="left")

# buildings_group_list = [i for i in set(buildings_geo[group_field]) if i not in ["other", "other"]]

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



# ---------------------------------------------------------
# spatialite approach for length of roads
# ---------------------------------------------------------

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



# ---------------------------------------------------------
# spatialite approach for nearest road
# ---------------------------------------------------------

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

#     knn_task_list = [gen_knn_query_string(i[geom_id], i.centroid_wkt, type_list, sqlite_road_table_name) for _,i in buffers_gdf.iterrows()]

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

