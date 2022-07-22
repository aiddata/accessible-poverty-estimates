
import os
import datetime
import sqlite3

import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

import prefect
from prefect import task

from utils import SpatiaLite


@task
def init_spatialite(sqlite_path, sqlite_lib_path):
    SL = SpatiaLite(sqlite_path, sqlite_lib_path)
    return SL


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

    gdf = pd.concat([gpd.read_file(p) for p in paths if os.path.exists(p)])
    return gdf


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
def get_spatialite_groups(type_crosswalk_df, group_field, sqlite_access, table_name, osm_type_field):
    sqlite_conn = sqlite_access.build_connection()
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
def run_sqlite_query(task, sqlite_access):
    sqlite_conn = sqlite_access.build_connection()
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
def create_aggegate_metrics(query_gdf_output, osm_group_lists, osm_type, require_all_groups=False):
    # use group results to generate "all" results
    group = 'all'
    group_list = osm_group_lists['group']
    if osm_type == 'buildings':
        if not require_all_groups:
            group_list = [
                g for g in group_list
                if f"{g}_{osm_type}_count" in query_gdf_output.columns
                and f"{g}_{osm_type}_totalarea" in query_gdf_output.columns
            ]

        query_gdf_output[f"{group}_{osm_type}_count"] = query_gdf_output[[f"{g}_{osm_type}_count" for g in group_list]].sum(axis=1)
        query_gdf_output[f"{group}_{osm_type}_totalarea"] = query_gdf_output[[f"{g}_{osm_type}_totalarea" for g  in group_list]].sum(axis=1)

        query_gdf_output[f"{group}_{osm_type}_avgarea"] = query_gdf_output[f"{group}_{osm_type}_totalarea"] / query_gdf_output[f"{group}_{osm_type}_count"]
        query_gdf_output[f"{group}_{osm_type}_ratio"] = query_gdf_output[f"{group}_{osm_type}_totalarea"] / query_gdf_output['buffer_area']

    elif osm_type == 'roads':
        if not require_all_groups:
            group_list = [
                g for g in group_list
                if f"{g}_{osm_type}_length" in query_gdf_output.columns
            ]

        query_gdf_output[f"{group}_{osm_type}_length"] = query_gdf_output[[f"{g}_{osm_type}_length" for g in group_list]].sum(axis=1)

    elif osm_type == 'nearest':
        if not require_all_groups:
            group_list = [
                g for g in group_list
                if f"{g}_roads_nearestdist" in query_gdf_output.columns
            ]

        query_gdf_output[f"{group}_roads_nearestdist"] = query_gdf_output[[f"{g}_roads_nearestdist" for g in group_list]].min(axis=1)

        field_with_min_val = query_gdf_output[[f"{g}_roads_nearestdist" for g in group_list]].idxmin(axis=1)
        query_gdf_output[f"{group}_roads_nearestid"] = [query_gdf_output.loc[ix, osm_id] for ix, osm_id in field_with_min_val.apply(lambda x: x.replace('nearestdist', 'nearestid')).iteritems()]

    else:
        raise ValueError(f'Unknown osm_type: {osm_type}')

    return query_gdf_output


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def export_sqlite(query_gdf_output, features_path, osm_type, geom_id):
    query_gdf_output[geom_id] = query_gdf_output.index
    cols = [geom_id] + [i for i in query_gdf_output.columns if f'_{osm_type}_' in i]
    features = query_gdf_output[cols].copy(deep=True)
    features.fillna(0, inplace=True)
    features.to_csv(features_path, index=False, encoding="utf-8")


@task
def find_nearest(group_list, group_field, osm_gdf, query_gdf):

    query_gdf = query_gdf.copy(deep=True)
    src_points = query_gdf.apply(lambda x: (x.longitude, x.latitude), axis=1).to_list()

    results = []

    for group in group_list:

        print(f'Roads nearest ({group})')

        # subset based on group
        if group != "all":
            sub_osm_gdf = osm_gdf.loc[osm_gdf[group_field] == group].reset_index().copy(deep=True)

        # generate list of all road vertices and convert to geodataframe
        line_xy = sub_osm_gdf.apply(lambda x: (x.osm_id, x.geometry.xy), axis=1)
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

        results.append(query_gdf[["{}_roads_nearestid".format(group), "{}_roads_nearestdist".format(group)]])

    return results


@task
def get_group_list(x):
    return x['group'].to_list()


@task
def flow_print(x):
    print(x)


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def merge_road_nearest_features_data(gdf, group_df_list):
    # print(gdf)
    for df in group_df_list:
        # merge group columns back to main cluster dataframe
        gdf = gdf.merge(df, left_index=True, right_index=True)
        # print(gdf)
    return gdf


@task
def load_geodataframe(path):
    return gpd.read_file(path)


@task
def merge_road_features(x, y):
    x = x[['DHSID'] + [i for i in x.columns if 'roads' in i]]
    y = y[['DHSID'] + [i for i in y.columns if 'roads' in i]]
    gdf = x.merge(y, on='DHSID')
    gdf.set_index('DHSID', inplace=True)
    return gdf


@task(log_stdout=True, max_retries=5, retry_delay=datetime.timedelta(seconds=10))
def export_road_features(gdf, geom_id, osm_type, path):
    # output final features
    gdf[geom_id] = gdf.index
    feature_cols = [i for i in gdf.columns if f"_{osm_type}_" in i]
    cols = [geom_id] + feature_cols
    features = gdf[cols].copy(deep=True)
    features.to_csv(path, index=False, encoding="utf-8")



