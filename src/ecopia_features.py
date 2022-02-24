'''
Extract features for machine learning models from Ecopia building and road footprint data

# convert shapefiles to spatialite databases first
# all files are in 4326, utm codes in original file names just indicate utm zone splits of data

cd /home/userx/Desktop/ecopia_data/
ogr2ogr -f SQLite -dsco SPATIALITE=YES africa_ghana_road.sqlite africa_ghana_road_4326/africa_ghana_road_4326.shp
# merge multiple building files into one
ogr2ogr -f SQLite -dsco SPATIALITE=YES africa_ghana_building.sqlite africa_ghana_building_32630/africa_ghana_building_32630.shp
ogr2ogr -f SQLite -update -append africa_ghana_building.sqlite africa_ghana_building_32631/africa_ghana_building_32631.shp -nln merge

'''

from pathlib import Path
import sqlite3

import pandas as pd
import geopandas as gpd


country = 'ghana'
country_utm_espg_code = 32630

group = "ecopia"

base_path = Path(f'/home/userx/Desktop/accessible-poverty-estimates/data/ecopia_data/{country}')

ecopia_building_path = base_path / f'africa_{country}_building.sqlite'
ecopia_road_path = base_path / f'africa_{country}_road.sqlite'

building_table_name = 'africa_ghana_building_32630'
road_table_name = 'africa_ghana_road_4326'

geom_label = 'dhs-buffers'
geom_id = 'DHSID'
dhs_path = '/home/userx/Desktop/PHL_WORK/osm-development-estimates/data/outputs/GH_2014_DHS/dhs_buffers.geojson'

buffers_gdf = gpd.read_file(dhs_path)
buffers_gdf = buffers_gdf.set_crs(epsg=4326)
buffers_gdf['buffer'] = buffers_gdf['geometry']
buffers_gdf = buffers_gdf.to_crs(epsg=country_utm_espg_code)
buffers_gdf['geometry'] = buffers_gdf.centroid
buffers_gdf = buffers_gdf.to_crs(epsg=4326)
buffers_gdf['centroid_wkt'] = buffers_gdf.geometry.apply(lambda x: x.wkt)
buffers_gdf['geometry'] = buffers_gdf['buffer']
buffers_gdf = buffers_gdf.drop(columns=['buffer'])


# copy of buffers gdf to use for outputs
buffers_gdf_buildings = buffers_gdf.copy(deep=True)
buffers_gdf_roads = buffers_gdf.copy(deep=True)


# =============================================================================

# add steps to build latest spatialite from source?
#

# spatialite info:
#   https://www.gaia-gis.it/fossil/libspatialite/index


def build_gpkg_connection(sqlite_path):

    # create connection and load spatialite extension
    conn = sqlite3.connect(sqlite_path)

    # enable SpatialLite extension
    conn.enable_load_extension(True)

    # to find existing path:
    # > whereis mod_spatialite.so
    # was version 4.3.0
    # default_shared_lib = '/usr/lib/x86_64-linux-gnu/mod_spatialite.so'

    # built v5.0.1 from source
    custom_shared_lib = '/usr/local/lib/mod_spatialite.so'

    # could not get libspatialite working either via apt install or conda install
    # libspatialite_path = os.path.join(os.environ["CONDA_PREFIX"], 'lib/libspatialite.so')


    # conn.load_extension(default_shared_lib)
    conn.load_extension(custom_shared_lib)
    # conn.load_extension(libspatialite_path)

    # initialise spatial table support
    conn.execute('SELECT InitSpatialMetadata(1)')
    conn.execute('SELECT CreateMissingSystemTables(1)')
    conn.execute('SELECT EnableGpkgAmphibiousMode()')

    # conn.execute('SELECT CreateSpatialIndex("africa_ghana_road_4326", "geom");')


    conn.execute("SELECT CreateMissingSystemTables(1);").fetchall()
    try:
        conn.execute("CREATE VIRTUAL TABLE KNN2 USING VirtualKNN2();")
    except:
        pass

    return conn


# =============================================================================
# road metrics

sqlite_road_conn = build_gpkg_connection(ecopia_road_path)
# sqlite_road_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
# sqlite_road_conn.execute(f'PRAGMA table_info({road_table_name})').fetchall()


# -------------------------------------
# distance to nearest road


def find_nearest(wkt):
    results = sqlite_road_conn.execute(f'SELECT fid,distance_m  FROM KNN2 WHERE f_table_name = "{road_table_name}" AND ref_geometry = PointFromText("{wkt}") AND radius=100.0 AND max_items=1').fetchall()
    return results

# took hours to run
# serial processing
nearest_road_data = buffers_gdf.centroid_wkt.apply(lambda wkt: find_nearest(wkt))

# parallel processing
#

nearest_road_df = pd.DataFrame([i[0] for i in nearest_road_data], columns=[f"{group}_roads_nearestid", f"{group}_roads_nearestdist"])

buffers_gdf_roads = pd.concat([buffers_gdf_roads, nearest_road_df], axis=1)


# -------------------------------------
# length of roads in buffer

buffers_gdf_roads[f"{group}_roads_length"] = None

# pretty fast
# serial processing
for _, row in buffers_gdf_roads.iterrows():

    int_wkt = row['geometry'].wkt

    q = f'''
        SELECT ogc_fid, AsText(st_intersection(geometry, GeomFromText("{int_wkt}"))) AS geometry
        FROM {road_table_name}
        WHERE st_intersects(geometry, GeomFromText("{int_wkt}"))
        '''

    df = pd.read_sql(q, sqlite_road_conn)
    df['geometry'] = gpd.GeoSeries.from_wkt(df.geometry)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=country_utm_espg_code)

    gdf['length'] = gdf.length
    total_length = gdf.length.sum()

    buffers_gdf_roads.loc[buffers_gdf_roads[geom_id] == row[geom_id], f"{group}_roads_length"] = total_length


buffers_gdf_roads[f"{group}_roads_length"].fillna(0, inplace=True)


sqlite_road_conn.close()


roads_cols = [geom_id, f"{group}_roads_nearestid", f"{group}_roads_nearestdist", f"{group}_roads_length"]
roads_features = buffers_gdf_roads[roads_cols].copy(deep=True)


roads_features_path = base_path / f'{country}_{geom_label}_{group}_roads.csv'
roads_features.to_csv(roads_features_path, index=False)



# =============================================================================
# building metrics

sqlite_building_conn = build_gpkg_connection(ecopia_building_path)
# sqlite_building_conn.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table'").fetchall()
# sqlite_building_conn.execute(f'PRAGMA table_info({building_table_name})').fetchall()


buffers_gdf_buildings[f"{group}_buildings_count"] = None
buffers_gdf_buildings[f"{group}_buildings_totalarea"] = None
buffers_gdf_buildings[f"{group}_buildings_avgarea"] = None
buffers_gdf_buildings[f"{group}_buildings_ratio"] = None

# took about an hour
# serial processing
for _, row in buffers_gdf_buildings.iterrows():

    int_wkt = row['geometry'].wkt

    q = f'''
        SELECT ogc_fid, AsText(st_intersection(geometry, GeomFromText("{int_wkt}"))) AS geometry
        FROM {building_table_name}
        WHERE st_intersects(geometry, GeomFromText("{int_wkt}"))
        '''

    df = pd.read_sql(q, sqlite_building_conn)
    df['geometry'] = gpd.GeoSeries.from_wkt(df.geometry)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=country_utm_espg_code)

    gdf['length'] = gdf.length
    total_length = gdf.length.sum()

    buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row[geom_id], f"{group}_buildings_count"] = gdf.shape[0]
    buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row[geom_id], f"{group}_buildings_totalarea"] = gdf.area.sum()
    buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row[geom_id], f"{group}_buildings_avgarea"] = gdf.area.mean()
    buffers_gdf_buildings.loc[buffers_gdf_buildings[geom_id] == row[geom_id], f"{group}_buildings_ratio"] = gdf.area.sum() / row.geometry.area



sqlite_building_conn.close()


buildings_cols = [geom_id, f"{group}_buildings_count", f"{group}_buildings_totalarea", f"{group}_buildings_avgarea", f"{group}_buildings_ratio"]
buildings_features = buffers_gdf_buildings[buildings_cols].copy(deep=True)
buildings_features.fillna(0, inplace=True)


buildings_features_path = base_path / f'{country}_{geom_label}_{group}_buildings.csv'
buildings_features.to_csv(buildings_features_path, index=False)
