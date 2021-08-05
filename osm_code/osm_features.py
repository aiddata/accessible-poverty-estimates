"""
python 3.8

portions of code and/or methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Extract features features OSM data

download OSM data from
http://download.geofabrik.de/asia/philippines.html#


buildings (polygons)
types : residential, damaged, commercial, industrial, education, health
For each type, we calculated
    - the total number of buildings (count poly features intersecting with buffer)
    - the total area of buildings (sum of area of poly features which intersect with buffer)
    - the mean area of buildings (avg area of poly features which intersect with buffer)
    - the proportion of the cluster area occupied by the buildings (ratio of total area of buildings which intersect with buffer to buffer area)

pois (points)
types: 100+ different types
For each type, we calculated
    - the total number of each POI within a proximity of the area (point in poly)

roads (lines)
types: primary, trunk, paved, unpaved, intersection
for each type of road, we calculated
    - the distance to the closest road (point to line vertice dist)
    - total number of roads (count line features which intersect with buffer)
    - total road length (length of lines which intersect with buffer)

"""

import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from sklearn.neighbors import BallTree
import numpy as np

project_dir = "/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK"
data_dir = os.path.join(project_dir, 'data')

date = "210101"


# >>>>>>>>>>>>>>>>>>>>
# DHS CLUSTERS

geom_label = "dhs-buffers"
geom_path = os.path.join(data_dir, 'dhs_buffers.geojson')
geom_id = "DHSID"

# load buffers/geom created during data prep
buffers_gdf = gpd.read_file(geom_path)

# calculate area of each buffer
# convert to UTM 51N (meters) first, then back to WGS84 (degrees)
buffers_gdf = buffers_gdf.to_crs("EPSG:32651") # UTM 51N
buffers_gdf["buffer_area"] = buffers_gdf.area
buffers_gdf = buffers_gdf.to_crs("EPSG:4326") # WGS84

# >>>>>>>>>>>>>>>>>>>>
# KC CLUSTERS

# geom_label = "kc-5km-buffers"
# geom_path = os.path.join(data_dir, 'kc_clusters_5km-buffer.geojson')
# geom_id = "cluster_name"

# # load point geom created during prep
# buffers_gdf = gpd.read_file(geom_path)
# buffers_gdf.columns = [i if i != "cluster" else "cluster_name" for i in buffers_gdf.columns]

# # calculate area of each buffer
# # convert to UTM 51N (meters) first, then back to WGS84 (degrees)
# buffers_gdf = buffers_gdf.to_crs("EPSG:32651") # UTM 51N
# buffers_gdf["buffer_area"] = buffers_gdf.area
# buffers_gdf = buffers_gdf.to_crs("EPSG:4326") # WGS84

# >>>>>>>>>>>>>>>>>>>>
# OSM PLACES

# geom_label = "osm-places-3km-buffers"
# geom_path = os.path.join(data_dir, 'osm-places_3km-buffer_{}.geojson'.format(date))
# geom_id = "osm_id"

# # load buffers/geom created during data prep
# buffers_gdf = gpd.read_file(geom_path)

# # calculate area of each buffer
# # convert to UTM 51N (meters) first, then back to WGS84 (degrees)
# buffers_gdf = buffers_gdf.to_crs("EPSG:32651") # UTM 51N
# buffers_gdf["buffer_area"] = buffers_gdf.area
# buffers_gdf = buffers_gdf.to_crs("EPSG:4326") # WGS84

# >>>>>>>>>>>>>>>>>>>>


# ---------------------------------------------------------
# pois
# count of each type of pois (100+) in each buffer

print("Running pois...")

osm_pois_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_pois_free_1.shp'.format(date))
osm_pois_a_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_pois_a_free_1.shp'.format(date))

raw_pois_geo = gpd.read_file(osm_pois_shp_path)
raw_pois_a_geo = gpd.read_file(osm_pois_a_shp_path)

pois_geo = pd.concat([raw_pois_geo, raw_pois_a_geo])

"""
# manually generate crosswalk
#   first prep CSV with all types - can combine multiple OSM timesteps (see below)
#   then in Excel/whatever, assign group to each type/fclass

type_df = pd.DataFrame({"type": list(set(pois_geo["fclass"]))})
type_df["group"]= 0
type_df.to_csv(os.path.join(project_dir, "OSM/crosswalks/pois_type_crosswalk.csv"), index=False, encoding="utf-8")

"""

# load crosswalk for types and assign any not grouped to "other"
pois_type_crosswalk_path = os.path.join(project_dir, 'OSM/osm_code/crosswalks/pois_type_crosswalk.csv')
pois_type_crosswalk_df = pd.read_csv(pois_type_crosswalk_path)
pois_type_crosswalk_df.loc[pois_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
pois_geo = pois_geo.merge(pois_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

pois_geo.loc[pois_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(pois_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# pois_group_list = ["all"] + [i for i in set(pois_geo[group_field])]
pois_group_list = [i for i in set(pois_geo[group_field])]

# copy of buffers gdf to use for output
buffers_gdf_pois = buffers_gdf.copy(deep=True)

for group in pois_group_list:
    print(group)
#     subet by group
    if group == "all":
        pois_geo_subset = pois_geo.reset_index(inplace=True).copy(deep=True)
    else:
        pois_geo_subset = pois_geo.loc[pois_geo[group_field] == group].reset_index().copy(deep=True)
#     query to find pois in each buffer
    bquery = pois_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
    # pois dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "pois": bquery[1]})
#     add pois data to spatial query dataframe
    bquery_full = bquery_df.merge(pois_geo_subset, left_on="pois", right_index=True, how="left")
#     aggregate spatial query df with pois info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({"pois": "count"})
    bquery_agg.columns = [group + "_pois_count"]
#     join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
#     not each cluster will have relevant pois, set those to zero
    z1.fillna(0, inplace=True)
#     set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[group + "_pois_count"]
#     merge group columns back to main cluster dataframe
    buffers_gdf_pois = buffers_gdf_pois.merge(z2, left_index=True, right_index=True)

# output final features
pois_feature_cols = [geom_id] + [i for i in buffers_gdf_pois.columns if "_pois_" in i]
pois_features = buffers_gdf_pois[pois_feature_cols].copy(deep=True)
pois_features_path = os.path.join(data_dir, 'osm/features/{}_pois_{}.csv'.format(geom_label, date))
pois_features.to_csv(pois_features_path, index=False, encoding="utf-8")


# ---------------------------------------------------------
# traffic
# count of each type of traffic item in each buffer

print("Running traffic...")

osm_traffic_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_traffic_free_1.shp'.format(date))
osm_traffic_a_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_traffic_a_free_1.shp'.format(date))

raw_traffic_geo = gpd.read_file(osm_traffic_shp_path)
raw_traffic_a_geo = gpd.read_file(osm_traffic_a_shp_path)

traffic_geo = pd.concat([raw_traffic_geo, raw_traffic_a_geo])


"""
# manually generate crosswalk
#   first prep CSV with all types - can combine multiple OSM timesteps (see below)
#   then in Excel/whatever, assign group to each type/fclass

type_df = pd.DataFrame({"type": list(set(traffic_geo["fclass"]))})
type_df["group"]= 0
type_df.to_csv(os.path.join(project_dir, "OSM/crosswalks/traffic_type_crosswalk.csv"), index=False, encoding="utf-8")

"""

# load crosswalk for types and assign any not grouped to "other"
traffic_type_crosswalk_path = os.path.join(project_dir, 'OSM/osm_code/crosswalks/traffic_type_crosswalk.csv')
traffic_type_crosswalk_df = pd.read_csv(traffic_type_crosswalk_path)
traffic_type_crosswalk_df.loc[traffic_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
traffic_geo = traffic_geo.merge(traffic_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

traffic_geo.loc[traffic_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(traffic_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# traffic_group_list = ["all"] + [i for i in set(traffic_geo[group_field])]
traffic_group_list = [i for i in set(traffic_geo[group_field])]

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
traffic_feature_cols = [geom_id] + [i for i in buffers_gdf_traffic.columns if "_traffic_" in i]
traffic_features = buffers_gdf_traffic[traffic_feature_cols].copy(deep=True)
traffic_features_path = os.path.join(data_dir, 'osm/features/{}_traffic_{}.csv'.format(geom_label, date))
traffic_features.to_csv(traffic_features_path, index=False, encoding="utf-8")

# ---------------------------------------------------------
# transport
# count of each type of transport item in each buffer

print("Running transport...")

osm_transport_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_transport_free_1.shp'.format(date))
osm_transport_a_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_transport_a_free_1.shp'.format(date))

raw_transport_geo = gpd.read_file(osm_transport_shp_path)
raw_transport_a_geo = gpd.read_file(osm_transport_a_shp_path)

transport_geo = pd.concat([raw_transport_geo, raw_transport_a_geo])


"""
 manually generate crosswalk
   first prep CSV with all types - can combine multiple OSM timesteps (see below)
   then in Excel/whatever, assign group to each type/fclass

type_df = pd.DataFrame({"type": list(set(transport_geo["fclass"]))})
type_df["group"]= 0
type_df.to_csv(os.path.join(project_dir, "OSM/crosswalks/transport_type_crosswalk.csv"), index=False, encoding="utf-8")

"""

# load crosswalk for types and assign any not grouped to "other"
transport_type_crosswalk_path = os.path.join(project_dir, 'OSM/osm_code/crosswalks/transport_type_crosswalk.csv')
transport_type_crosswalk_df = pd.read_csv(transport_type_crosswalk_path)
transport_type_crosswalk_df.loc[transport_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
transport_geo = transport_geo.merge(transport_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

transport_geo.loc[transport_geo["fclass"].isna(), "group"] = "unclassified"

# show breakdown of groups
print(transport_geo.group.value_counts())

# group_field = "fclass"
group_field = "group"

# split by group
# transport_group_list = ["all"] + [i for i in set(transport_geo[group_field])]
transport_group_list = [i for i in set(transport_geo[group_field])]

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
transport_feature_cols = [geom_id] + [i for i in buffers_gdf_transport.columns if "_transport_" in i]
transport_features = buffers_gdf_transport[transport_feature_cols].copy(deep=True)
transport_features_path = os.path.join(data_dir, 'osm/features/{}_transport_{}.csv'.format(geom_label, date))
transport_features.to_csv(transport_features_path, index=False, encoding="utf-8")


# ---------------------------------------------------------
# # buildings
# # for each type of building (and all buildings combined)
# # count of buildings in each buffer, average areas of buildings in each buffer, total area of building in each buffer, ratio of building area to total area of buffer

print("Running buildings...")

osm_buildings_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_buildings_a_free_1.shp'.format(date))
buildings_geo_raw = gpd.read_file(osm_buildings_shp_path)

"""
 manually generate crosswalk
 first prep CSV with all types - can combine multiple OSM timesteps (see below)
 then in Excel/whatever, assign group to each type/fclass

 type_df = pd.DataFrame({"type": list(set(buildings_geo["type"]))})
 type_df["group"]= 0
 type_df.to_csv(os.path.join(project_dir, "OSM/crosswalks/building_type_crosswalk.csv"), index=False, encoding="utf-8")

 """

# # load crosswalk for building types and assign any not grouped to "other"
building_type_crosswalk_path = os.path.join(project_dir, 'OSM/osm_code/osm_code/crosswalks/building_type_crosswalk.csv')
building_type_crosswalk_df = pd.read_csv(building_type_crosswalk_path)
building_type_crosswalk_df.loc[building_type_crosswalk_df["group"] == "0", "group"] = "other"

# # merge new classification and assign any buildings without a type to unclassifid
buildings_geo_raw = buildings_geo_raw.merge(building_type_crosswalk_df, on="type", how="left")

buildings_geo_raw.loc[buildings_geo_raw["type"].isna(), "group"] = "unclassified"

group_field = "group"

# # show breakdown of groups
print(buildings_geo_raw.group.value_counts())

buildings_geo = buildings_geo_raw.copy(deep=True)

# # split by building types
# # group_list = ["residential"]
# # group_list = ["all"] + [i for i in set(buildings_geo["group"]) if i not in ["other", "unclassified"]]
buildings_group_list = [i for i in set(buildings_geo["group"]) if i not in ["other", "unclassified"]]

buildings_group_list = [i for i in buildings_group_list if str(i) != 'nan']  #removes nan from building_group_list - Sasan

buildings_group_list = buildings_group_list + ['all'] #add a section for all buildings into group lost



if "all" not in buildings_group_list:
    buildings_geo = buildings_geo.loc[buildings_geo["group"].isin(buildings_group_list)]

# calculate area of each building
# convert to UTM 51N (meters) first, then back to WGS84 (degrees)
buildings_geo = buildings_geo.to_crs("EPSG:32651") # UTM 51N
buildings_geo["area"] = buildings_geo.area
buildings_geo = buildings_geo.to_crs("EPSG:4326") # WGS84


# copy of buffers gdf to use for output
buffers_gdf_buildings = buffers_gdf.copy(deep=True)

for group in buildings_group_list:
    print(group)
#     subet by group
    if group == "all":
        buildings_geo_subset = buildings_geo.copy(deep=True)
    else:
        buildings_geo_subset = buildings_geo.loc[buildings_geo[group_field] == group].reset_index().copy(deep=True)
#     query to find buildings in each buffer
    bquery = buildings_geo_subset.sindex.query_bulk(buffers_gdf.geometry)
#     building dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "building": bquery[1]})
#     add building data to spatial query dataframe
    bquery_full = bquery_df.merge(buildings_geo_subset, left_on="building", right_index=True, how="left")
#     aggregate spatial query df with building info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({
        "area": ["count", "mean", "sum"]
    })
#     rename agg df
    basic_building_cols = ["buildings_count", "buildings_avgarea", "buildings_totalarea"]
    bquery_agg.columns = ["{}_{}".format(group, i) for i in basic_building_cols]
#     join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
#     not each cluster will have relevant buildings, set those to zero
    z1.fillna(0, inplace=True)
#     calculate ratio for building type
    z1["{}_buildings_ratio".format(group)] = z1["{}_buildings_totalarea".format(group)] / z1["buffer_area"]
#     set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[bquery_agg.columns.to_list() + ["{}_buildings_ratio".format(group)]]
#     merge group columns back to main cluster dataframe
    buffers_gdf_buildings = buffers_gdf_buildings.merge(z2, left_index=True, right_index=True)


# output final features
buildings_feature_cols = [geom_id] + [i for i in buffers_gdf_buildings.columns if "_buildings_" in i]
buildings_features = buffers_gdf_buildings[buildings_feature_cols].copy(deep=True)
buildings_features_path = os.path.join(data_dir, 'osm/features/{}_buildings_{}.csv'.format(geom_label, date))
buildings_features.to_csv(buildings_features_path, index=False, encoding="utf-8")

# ---------------------------------------------------------
# roads
# for each type of road
# distance to closest road from cluster centroid, total number of roads in each cluster, and total length of roads in each cluster

print("Running roads...")

osm_roads_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_roads_free_1.shp'.format(date))
roads_geo = gpd.read_file(osm_roads_shp_path)

# get each road length
# convert to UTM 51N (meters) first, then back to WGS84 (degrees)
roads_geo = roads_geo.to_crs("EPSG:32651") # UTM 51N
roads_geo["road_length"] = roads_geo.geometry.length
roads_geo = roads_geo.to_crs("EPSG:4326") # WGS84


"""
# manually generate crosswalk
#   first prep CSV with all types - can combine multiple OSM timesteps (see below)
#   then in Excel/whatever, assign group to each type/fclass

type_df = pd.DataFrame({"type": list(set(roads_geo["fclass"]))})
type_df["group"]= 0
type_df.to_csv(os.path.join(project_dir, "OSM/crosswalks/roads_type_crosswalk.csv"), index=False, encoding="utf-8")

"""

# load crosswalk for types and assign any not grouped to "other"
roads_type_crosswalk_path = os.path.join(project_dir, 'OSM/osm_code/crosswalks/roads_type_crosswalk.csv')
roads_type_crosswalk_df = pd.read_csv(roads_type_crosswalk_path)
roads_type_crosswalk_df.loc[roads_type_crosswalk_df["group"] == "0", "group"] = "other"

# merge new classification and assign any features without a type to unclassifid
roads_geo = roads_geo.merge(roads_type_crosswalk_df, left_on="fclass", right_on="type", how="left")

roads_geo.loc[roads_geo["fclass"].isna(), "group"] = "unclassified"

# group_field = "fclass"
group_field = "group"

# show breakdown of groups
print(roads_geo[group_field].value_counts())


# split by groups
roads_group_list = [i for i,j in roads_geo[group_field].value_counts().to_dict().items() if j > 1000]
# roads_group_list = ["all"] + [i for i,j in roads_geo[group_field].value_counts().to_dict().items() if j > 1000]
# roads_group_list = ["all"] + [i for i in set(roads_geo["fclass"])]
# roads_group_list = ["all", "primary", "secondary"]


#-----------------
#find distance to nearest road (based on vertices of roads)


# generate centroids of buffers
cluster_centroids = buffers_gdf.copy(deep=True)
cluster_centroids.geometry = cluster_centroids.apply(lambda x: Point(x.longitude, x.latitude), axis=1)
cluster_centroids = gpd.GeoDataFrame(cluster_centroids)

src_points = cluster_centroids.apply(lambda x: (x.longitude, x.latitude), axis=1).to_list()


for group in roads_group_list:
    print(group)
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
    cluster_centroids["{}_roads_nearest-osmid".format(group)] = list(map(osm_id_lookup, closest))
    cluster_centroids["{}_roads_nearestdist".format(group)] = closest_dist



cluster_centroids = cluster_centroids[[geom_id] + [i for i in cluster_centroids.columns if "_roads_" in i]]
cluster_centroids.set_index(geom_id, inplace=True)


# # -----------------
# # calculate number of roads and length of roads intersecting with each buffer

# # copy of buffers gdf to use for output
buffers_gdf_roads = buffers_gdf.copy(deep=True)

for group in roads_group_list:
    print(group)
    if group == "all":
        subset_roads_geo = roads_geo.copy(deep=True)
    else:
        subset_roads_geo = roads_geo.loc[roads_geo[group_field] == group].reset_index().copy(deep=True)
    # query to find roads in each buffer
    bquery = subset_roads_geo.sindex.query_bulk(buffers_gdf.geometry)
    # roads dataframe where each column contains a cluster and one building in it (can have multiple rows per cluster)
    bquery_df = pd.DataFrame({"cluster": bquery[0], "roads": bquery[1]})
    # add roads data to spatial query dataframe
    bquery_full = bquery_df.merge(roads_geo, left_on="roads", right_index=True, how="left")
    # aggregate spatial query df with roads info, by cluster
    bquery_agg = bquery_full.groupby("cluster").agg({"road_length": ["count", "sum"]})
    bquery_agg.columns = [group + "_roads_count", group + "_roads_length"]
    # join cluster back to original buffer_geo dataframe with columns for specific building type queries
    z1 = buffers_gdf.merge(bquery_agg, left_index=True, right_on="cluster", how="left")
    # not each cluster will have relevant roads, set those to zero
    z1.fillna(0, inplace=True)
    # set index and drop unnecessary columns
    if z1.index.name != "cluster": z1.set_index("cluster", inplace=True)
    z2 = z1[[group + "_roads_count", group + "_roads_length"]]
    # merge group columns back to main cluster dataframe
    buffers_gdf_roads = buffers_gdf_roads.merge(z2, left_index=True, right_index=True)


# output final features
roads_features = buffers_gdf_roads.merge(cluster_centroids, on=geom_id)
roads_feature_cols = [geom_id] + [i for i in roads_features.columns if "_roads_" in i]
roads_features = roads_features[roads_feature_cols].copy(deep=True)
roads_features_path = os.path.join(data_dir, 'osm/features/{}_roads_{}.csv'.format(geom_label, date))
roads_features.to_csv(roads_features_path, index=False, encoding="utf-8")

# ------------------------
# Compute and then aggregate data on all roads count, length, and nearest distances, later adding them to the roads_features csv

#read in previously created osm_road features data frame
roads_df = pd.read_csv('/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK/data/osm/features/dhs-buffers_roads_210101.csv')

#create lists to keep track of aggregated road data

allroads_count =  []
allroads_length = []
allroads_nearestdist = []

#iterate through each row (cluster), and aggregate the road count and length & find the minimum nearest road

for cluster, series_vals in roads_df.iterrows():
    count = 0
    length = 0
    nearest_dist = []
    for col_name, val in series_vals.iteritems():
        if 'count' in col_name:
            count += val
        elif 'length' in col_name:
            length += val
        elif 'dist' in col_name:
            nearest_dist.append(val)
    allroads_count.append(count)
    allroads_length.append(length)
    allroads_nearestdist.append(np.min(nearest_dist))

roads_df['all_roads_count'] = allroads_count
roads_df['all_roads_nearestdist'] = allroads_nearestdist
roads_df['all_roads_length'] = allroads_length

roads_df_path = os.path.join(data_dir, 'osm/features/{}_roads_{}.csv'.format(geom_label, date))
roads_df.to_csv(roads_df_path, index=False, encoding="utf-8")


