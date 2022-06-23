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

import prefect
from prefect import task, Flow, Client, unmapped
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor
from prefect.run_configs import LocalRun


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]
data_dir = os.path.join(project_dir, 'data')

dask_enabled = config["main"]["dask_enabled"]
prefect_cloud_enabled = config["main"]["prefect_cloud_enabled"]


# ---------------------------------------------------------


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def load_dhs_data(geom_path, geom_id, country_utm_epsg_code):

    # load buffers/geom created during data prep
    buffers_gdf = gpd.read_file(geom_path)

    # calculate area of each buffer
    # convert to UTM first, then back to WGS84 (degrees)
    buffers_gdf = buffers_gdf.to_crs(epsg=country_utm_epsg_code)
    buffers_gdf["buffer_area"] = buffers_gdf.area
    buffers_gdf = buffers_gdf.to_crs("EPSG:4326") # WGS84
    buffers_gdf['longitude'] = buffers_gdf.centroid.x
    buffers_gdf['latitude'] = buffers_gdf.centroid.y
    buffers_gdf['centroid_wkt'] = buffers_gdf.geometry.centroid.apply(lambda x: x.wkt)

    buffers_gdf.set_index(geom_id, inplace=True)
    return buffers_gdf


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def load_crosswalk(type, data_dir):
    # load crosswalk for types and assign any not grouped to "other"
    type_crosswalk_path = os.path.join(data_dir, f'crosswalks/{type}_type_crosswalk.csv')
    type_crosswalk_df = pd.read_csv(type_crosswalk_path)
    type_crosswalk_df.loc[type_crosswalk_df["group"] == "0", "group"] = "other"
    type_crosswalk_df = type_crosswalk_df.loc[type_crosswalk_df.type.notna()]
    return type_crosswalk_df


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def load_osm_shp(type, data_dir, country_name, osm_date):

    paths = [
        os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_{type}_free_1.shp'),
        os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_{type}_a_free_1.shp')
    ]

    gdf_list = [gpd.read_file(p) for p in paths if os.path.exists(p)]

    gdf = pd.concat(gdf_list)
    return gdf



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



@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def merge_crosswalk(osm_raw_gdf, type_crosswalk_df):
    # merge new classification and assign any features without a type to unclassifid
    geo_gdf = osm_raw_gdf.merge(type_crosswalk_df, left_on="fclass", right_on="type", how="left")
    geo_gdf.loc[geo_gdf["fclass"].isna(), "group"] = "other"
    return geo_gdf


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def get_groups(osm_gdf, group_field):
    # show breakdown of groups
    print(osm_gdf[group_field].value_counts())

    # split by group
    # group_list = ["all"] + [i for i in set(osm_gdf[group_field])]
    group_list = [i for i in set(osm_gdf[group_field]) if pd.notnull(i)]
    return group_list


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def get_spatialite_groups(type_crosswalk_df, group_field, sqlite_path, table_name, osm_type_field):
    sqlite_conn = build_gpkg_connection(sqlite_path)
    group_lists = type_crosswalk_df.groupby(group_field).agg({'type':list}).reset_index()
    print(group_lists)
    summary = ''
    for i in group_lists.itertuples():
        _, group, type_list = i
        if len(type_list) == 1:
            type_list.append('unused_random_string')
        q = f'''
            SELECT ogc_fid
            FROM {table_name}
            WHERE {osm_type_field} in {tuple(type_list)}
            '''
        r = pd.read_sql(q, sqlite_conn)
        summary += f'\n{group}: {len(r)}'
    print(summary)
    sqlite_conn.close()
    return group_lists


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def point_query(group, group_field, query_gdf, osm_gdf, osm_type):
    logger = prefect.context.get("logger")
    logger.info(f'{osm_type} {group} {group_field}')

    print(osm_type, ':', group)
    # copy of buffers gdf to use for output
    query_gdf = query_gdf.copy(deep=True)
    query_gdf_index = query_gdf.index.name
    query_gdf.reset_index(inplace=True)

    # subet by group
    if group == "all":
        osm_subset_gdf = osm_gdf.reset_index(inplace=True).copy(deep=True)
    else:
        osm_subset_gdf = osm_gdf.loc[osm_gdf[group_field] == group].reset_index().copy(deep=True)

    # query to find pois in each buffer
    bquery = osm_subset_gdf.sindex.query_bulk(query_gdf.geometry)
    # pois dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], osm_type: bquery[1]})
    # add pois data to spatial query dataframe
    bquery_full = bquery_df.merge(osm_subset_gdf, left_on=osm_type, right_index=True, how="left")
    # aggregate spatial query df with pois info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({osm_type: "count"})
    bquery_agg.columns = [group + f"_{osm_type}_count"]
    # join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = query_gdf.merge(bquery_agg, left_index=True, right_index=True, how="left")
    # not each cluster will have relevant pois, set those to zero
    z1.fillna(0, inplace=True)
    # set index and drop unnecessary columns
    z1.set_index(query_gdf_index, inplace=True)
    z2 = z1[group + f"_{osm_type}_count"]
    return z2


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def merge_features_data(gdf, group_df_list):
    for df in group_df_list:
        # merge group columns back to main cluster dataframe
        gdf = gdf.merge(df, left_index=True, right_index=True)
    return gdf


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def export_point_features(gdf, geom_id, osm_type, path):
    # output final features
    feature_cols = [i for i in gdf.columns if f"_{osm_type}_" in i]
    features = gdf[feature_cols].copy(deep=True)
    features[f'all_{osm_type}_count'] = features[feature_cols].sum(axis=1)
    features[geom_id] = features.index
    cols = [geom_id] + [i for i in features.columns if i != geom_id]
    features[cols].to_csv(path, index=False, encoding="utf-8")


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def create_sqlite_task_list(osm_group_lists, query_gdf, table_name, osm_type_field):

    task_list = []
    for i in osm_group_lists.itertuples():
        _, group, type_list = i

        for ix, (index, row) in enumerate(query_gdf.iterrows()):

            int_wkt = row['geometry'].wkt

            q = f'''
                SELECT ogc_fid, AsText(st_intersection(geometry, GeomFromText("{int_wkt}"))) AS geometry
                FROM {table_name}
                WHERE {osm_type_field} in {tuple(type_list)} AND st_intersects(geometry, GeomFromText("{int_wkt}"))
                '''

            task = (group, index, q)
            task_list.append(task)
            # if ix > 20:
            #     break
    return task_list


@task(log_stdout=True, max_retries=10, retry_delay=datetime.timedelta(seconds=10))
def run_sqlite_query(task, sqlite_path):
    sqlite_conn = build_gpkg_connection(sqlite_path)
    group, id, q = task
    df = pd.read_sql(q, sqlite_conn)
    sqlite_conn.close()
    return group, id, df


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def process_sqlite_results(results_list, query_gdf, country_utm_epsg_code, geom_id, osm_type):
    logger = prefect.context.get("logger")

    query_gdf_output = query_gdf.copy(deep=True)

    for group, row_index, result_df in results_list:
        logger.info(f'{group} {row_index}')

        df = result_df.copy(deep=True)
        df['geometry'] = gpd.GeoSeries.from_wkt(df.geometry)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(epsg=country_utm_epsg_code)

        if osm_type == 'buildings':
            # print(query_gdf_output.loc[row_index, 'buffer_area'])

            buffer_area = query_gdf_output.loc[row_index, 'buffer_area']
            query_gdf_output.loc[row_index, f"{group}_buildings_count"] = gdf.shape[0]

            query_gdf_output.loc[row_index, f"{group}_buildings_totalarea"] = gdf.area.sum()
            query_gdf_output.loc[row_index, f"{group}_buildings_avgarea"] = gdf.area.mean()
            query_gdf_output.loc[row_index, f"{group}_buildings_ratio"] = gdf.area.sum() / buffer_area

        elif osm_type == 'roads':
            gdf['length'] = gdf.length
            total_length = gdf.length.sum()

            query_gdf_output.loc[row_index, f"{group}_roads_length"] = total_length

        else:
            raise ValueError(f'Unknown OSM type: {osm_type}')

    return query_gdf_output


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def create_aggegate_metrics(query_gdf_output, osm_group_lists, osm_type):
    # use group results to generate "all" results
    group = 'all'

    if osm_type == 'buildings':
        query_gdf_output[f"{group}_{osm_type}_count"] = query_gdf_output[[f"{g}_{osm_type}_count" for g in osm_group_lists['group']]].sum(axis=1)
        query_gdf_output[f"{group}_{osm_type}_totalarea"] = query_gdf_output[[f"{g}_{osm_type}_totalarea" for g  in osm_group_lists['group']]].sum(axis=1)
        query_gdf_output[f"{group}_{osm_type}_avgarea"] = query_gdf_output[f"{group}_{osm_type}_totalarea"] / query_gdf_output[f"{group}_{osm_type}_count"]
        query_gdf_output[f"{group}_{osm_type}_ratio"] = query_gdf_output[f"{group}_{osm_type}_totalarea"] / query_gdf_output['buffer_area']

    elif osm_type == 'roads':
        query_gdf_output[f"{group}_{osm_type}_length"] = query_gdf_output[[f"{g}_{osm_type}_length" for g  in osm_group_lists['group']]].sum(axis=1)

    elif osm_type == 'nearest':

        query_gdf_output[f"{group}_roads_nearestdist"] = query_gdf_output[[f"{g}_roads_nearestdist" for g  in osm_group_lists['group']]].min(axis=1)

        field_with_min_val = query_gdf_output[[f"{g}_roads_nearestdist" for g  in osm_group_lists['group']]].idxmin(axis=1)
        query_gdf_output[f"{group}_roads_nearestid"] = [query_gdf_output.loc[ix, osm_id] for ix, osm_id in field_with_min_val.apply(lambda x: x.replace('nearestdist', 'nearestid')).iteritems()]

    else:
        raise ValueError(f'Unknown osm_type: {osm_type}')

    return query_gdf_output


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def export_sqlite(query_gdf_output, features_path, osm_type):
    query_gdf_output[geom_id] = query_gdf_output.index
    cols = [geom_id] + [i for i in query_gdf_output.columns if f'_{osm_type}_' in i]
    features = query_gdf_output[cols].copy(deep=True)
    features.fillna(0, inplace=True)
    features.to_csv(features_path, index=False, encoding="utf-8")


@task
def find_nearest(group, group_field, osm_gdf, query_gdf, geom_id):

    print(f'Roads nearest ({group})')

    query_gdf = query_gdf.copy(deep=True)
    src_points = query_gdf.apply(lambda x: (x.longitude, x.latitude), axis=1).to_list()

    # subset based on group
    if group == "all":
        subset_osm_gdf = osm_gdf.copy(deep=True)
    else:
        subset_osm_gdf = osm_gdf.loc[osm_gdf[group_field] == group].reset_index().copy(deep=True)

    # generate list of all road vertices and convert to geodataframe
    line_xy = subset_osm_gdf.apply(lambda x: (x.osm_id, x.geometry.xy), axis=1)
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
    query_gdf["{}_roads_nearestid".format(group)] = list(map(osm_id_lookup, closest))
    query_gdf["{}_roads_nearestdist".format(group)] = closest_dist

    return query_gdf[["{}_roads_nearestid".format(group), "{}_roads_nearestdist".format(group)]]


@task
def simple(x):
    return x['group'].to_list()


@task
def flow_print(x):
    print(x)


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def merge_road_nearest_features_data(gdf, group_df_list):
    print(gdf)
    for df in group_df_list:
        # merge group columns back to main cluster dataframe
        gdf = gdf.merge(df, left_index=True, right_index=True)
        print(gdf)
    return gdf


@task
def merge_road_features(x, y):
    x = x[[i for i in x.columns if 'roads' in i]]
    y = y[[i for i in y.columns if 'roads' in i]]
    gdf = x.merge(y, left_index=True, right_index=True)
    return gdf


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def export_road_features(gdf, geom_id, osm_type, path):
    # output final features
    gdf[geom_id] = gdf.index
    feature_cols = [i for i in gdf.columns if f"_{osm_type}_" in i]
    cols = [geom_id] + feature_cols
    features = gdf[cols].copy(deep=True)
    features.to_csv(path, index=False, encoding="utf-8")





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
    buffers_gdf = load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    # ---------------------------------------------------------
    # pois
    # count of each type of pois (100+) in each buffer

    pois_geo_raw = load_osm_shp("pois", data_dir, country_name, osm_date)

    pois_type_crosswalk_df = load_crosswalk("pois", data_dir)

    pois_geo = merge_crosswalk(pois_geo_raw, pois_type_crosswalk_df)

    pois_group_list = get_groups(pois_geo, group_field)

    pois_group_data = point_query.map(pois_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(pois_geo), osm_type=unmapped("pois"))

    buffers_gdf_pois = merge_features_data(buffers_gdf, pois_group_data)

    pois_features_path = os.path.join(osm_features_dir, '{}_pois_{}.csv'.format(geom_label, osm_date))

    export_point_features(buffers_gdf_pois, geom_id, 'pois', pois_features_path)


    # ---------------------------------------------------------
    # traffic
    # count of each type of traffic item in each buffer

    traffic_geo_raw = load_osm_shp("traffic", data_dir, country_name, osm_date)

    traffic_type_crosswalk_df = load_crosswalk("traffic", data_dir)

    traffic_geo = merge_crosswalk(traffic_geo_raw, traffic_type_crosswalk_df)

    traffic_group_list = get_groups(traffic_geo, group_field)

    traffic_group_data = point_query.map(traffic_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(traffic_geo), osm_type=unmapped("traffic"))

    buffers_gdf_traffic = merge_features_data(buffers_gdf, traffic_group_data)

    traffic_features_path = os.path.join(osm_features_dir, '{}_traffic_{}.csv'.format(geom_label, osm_date))

    export_point_features(buffers_gdf_traffic, geom_id, 'traffic', traffic_features_path)


    # ---------------------------------------------------------
    # transport
    # count of each type of transport item in each buffer

    transport_geo_raw = load_osm_shp("transport", data_dir, country_name, osm_date)

    transport_type_crosswalk_df = load_crosswalk("transport", data_dir)

    transport_geo = merge_crosswalk(transport_geo_raw, transport_type_crosswalk_df)

    transport_group_list = get_groups(transport_geo, group_field)

    transport_group_data = point_query.map(transport_group_list, group_field=unmapped(group_field), query_gdf=unmapped(buffers_gdf), osm_gdf=unmapped(transport_geo), osm_type=unmapped("transport"))

    buffers_gdf_transport = merge_features_data(buffers_gdf, transport_group_data)

    transport_features_path = os.path.join(osm_features_dir, '{}_transport_{}.csv'.format(geom_label, osm_date))

    export_point_features(buffers_gdf_transport, geom_id, 'transport', transport_features_path)




with Flow("osm-features-buildings") as flow:

    building_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_buildings_a_free_1.sqlite')
    building_table_name = 'DATA_TABLE'
    osm_type_field = 'type'
    # sqlite_building_conn = build_gpkg_connection(building_path)
    # sqlite_building_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
    # sqlite_building_conn.execute(f'PRAGMA table_info({building_table_name})').fetchall()

    building_type_crosswalk_df = load_crosswalk("buildings", data_dir)

    building_group_lists = get_spatialite_groups(building_type_crosswalk_df, group_field, building_path, building_table_name, osm_type_field)

    geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
    buffers_gdf = load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    task_list = create_sqlite_task_list(building_group_lists, buffers_gdf, building_table_name, osm_type_field)

    # results_list = [run_sqlite_query(i, building_path) for i in task_list]
    results_list = run_sqlite_query.map(task_list, sqlite_path=unmapped(building_path))

    buffers_gdf_buildings = process_sqlite_results(results_list, buffers_gdf, country_utm_epsg_code, geom_id, 'buildings')

    buffers_gdf_buildings = create_aggegate_metrics(buffers_gdf_buildings, building_group_lists, 'buildings')

    buildings_features_path = os.path.join(osm_features_dir, f'{geom_label}_buildings_{osm_date}.csv')

    buildings_features = export_sqlite(buffers_gdf_buildings, buildings_features_path, 'buildings')




with Flow("osm-features-roads") as flow:

    road_path = os.path.join(data_dir, f'osm/{country_name}-{osm_date}-free.shp/gis_osm_roads_free_1.sqlite')
    road_table_name = 'DATA_TABLE'
    osm_type_field = 'fclass'

    road_type_crosswalk_df = load_crosswalk("roads", data_dir)

    road_group_lists = get_spatialite_groups(road_type_crosswalk_df, group_field, road_path, road_table_name, osm_type_field)

    geom_path = os.path.join(data_dir, 'outputs', output_name, 'dhs_buffers.geojson')
    buffers_gdf = load_dhs_data(geom_path, geom_id, country_utm_epsg_code)

    # ---------
    #length specific
    task_list = create_sqlite_task_list(road_group_lists, buffers_gdf, road_table_name, osm_type_field)

    # results_list = [run_sqlite_query(i, road_path) for i in task_list]
    results_list = run_sqlite_query.map(task_list, sqlite_path=unmapped(road_path))

    roads_length_gdf = process_sqlite_results(results_list, buffers_gdf, country_utm_epsg_code, geom_id, 'roads')

    roads_length_gdf = create_aggegate_metrics(roads_length_gdf, road_group_lists, 'roads')

    flow_print(roads_length_gdf)

    # ---------
    # nearest specific

    roads_geo_raw = load_osm_shp("roads", data_dir, country_name, osm_date)

    roads_geo = merge_crosswalk(roads_geo_raw, road_type_crosswalk_df)

    road_group_list = simple(road_group_lists)

    # nearest_results_list = [find_nearest(i, group_field, roads_geo, buffers_gdf, geom_id) for i in road_group_list]
    nearest_results_list = find_nearest.map(road_group_list, group_field=unmapped(group_field), osm_gdf=unmapped(roads_geo), query_gdf=unmapped(buffers_gdf), geom_id=unmapped(geom_id))

    roads_nearest_gdf = merge_road_nearest_features_data(buffers_gdf, nearest_results_list)

    # # create nearest aggregates
    roads_nearest_gdf = create_aggegate_metrics(roads_nearest_gdf, road_group_lists, 'nearest')

    flow_print(roads_nearest_gdf)

    # ---------


    # merge length gdf and nearest gdf
    roads_merge_gdf = merge_road_features(roads_length_gdf, roads_nearest_gdf)

    roads_features_path = os.path.join(osm_features_dir, '{}_roads_{}.csv'.format(geom_label, osm_date))
    export_road_features(roads_merge_gdf, geom_id, 'roads', roads_features_path)





if dask_enabled:
    # executor = DaskExecutor(address="tcp://127.0.0.1:8786")
    executor = LocalDaskExecutor(address="tcp://127.0.0.1:8786", scheduler="processes")
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

