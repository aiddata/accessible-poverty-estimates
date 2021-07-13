"""
Generates osm places dataset
"""
import os
import math
import pandas as pd
import geopandas as gpd
import rasterstats as rs

project_dir = "/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK"
data_dir = os.path.join(project_dir, 'data')

date = "210101"


# ---------------------------------------------------------
# places
# count of each type of places (100+) in each buffer

osm_places_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_places_free_1.shp'.format(date))
osm_places_a_shp_path = os.path.join(data_dir, 'osm/philippines-{}-free.shp/gis_osm_places_a_free_1.shp'.format(date))

raw_places_geo = gpd.read_file(osm_places_shp_path)
raw_places_a_geo = gpd.read_file(osm_places_a_shp_path)

gdf = pd.concat([raw_places_geo, raw_places_a_geo])

# drop island, region, country
valid_places_fclass_list = ['national_capital', 'town', 'city', 'suburb', 'village', 'hamlet', 'farm', 'locality']
gdf = gdf.loc[gdf.fclass.isin(valid_places_fclass_list)]

print(gdf.fclass.value_counts())


# convert to UTM 51N first (meters) then back to WGS84 (degrees)
gdf = gdf.to_crs("EPSG:32651") # UTM 51N
gdf.geometry = gdf.centroid
gdf = gdf.to_crs("EPSG:4326") # WGS84
gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y

gdf.index = pd.RangeIndex(0, len(gdf))


# save point geometry as geojson
points_gdf = gdf[["osm_id", "geometry", "longitude", "latitude"]]
points_path = os.path.join(data_dir, 'osm-places_points_{}.geojson'.format(date))
points_gdf.to_file(points_path, driver='GeoJSON')



# create buffer to use for finding osm features
buffer_gdf = gdf.copy(deep=True)
buffer_gdf = buffer_gdf.to_crs("EPSG:32651") # UTM 51N
buffer_gdf.geometry = buffer_gdf.buffer(3000)
buffer_gdf = buffer_gdf.to_crs("EPSG:4326") # WGS84

# save buffer geometry as geojson
buffer_gdf = buffer_gdf[["osm_id", "geometry", "longitude", "latitude"]]
buffer_path = os.path.join(data_dir, 'osm-places_3km-buffer_{}.geojson'.format(date))
buffer_gdf.to_file(buffer_path, driver='GeoJSON')



# ---------------------------------------------------------

# radius in meters
buffer_sizes = {
    "1km": 1000,
    "3km": 3000,
    "5km": 5000,
    "10km": 10000
}

for label, bsize in buffer_sizes.items():
    print(label)
    # generate buffer
    tmp_gdf = gdf.copy(deep=True)
    tmp_gdf = tmp_gdf.to_crs("EPSG:32651") # UTM 51N
    tmp_gdf.geometry = tmp_gdf.buffer(bsize)
    tmp_gdf = tmp_gdf.to_crs("EPSG:4326") # WGS84
    # get ntl zonal stats
    print("\tntl")
    ntl_path = os.path.join(data_dir, 'viirs/VNL_v2_npp_2016_global_vcmslcfg_c202101211500.average.tif')#clipped_VNL_v2_npp_2016_global_vcmslcfg_c202102150000_average_masked.tif')
    ntl_data = rs.zonal_stats(tmp_gdf.geometry, ntl_path, stats=["mean", "min", "max", "median", "sum"])
    ntl_df = pd.DataFrame(ntl_data)
    ntl_df.columns = ["ntl_{}_{}".format(label, i) for i in ntl_df.columns]
    # get pop zonal stats
    print("\tpop")
    pop_path = os.path.join(data_dir, 'population_phl_2018-10-01_geotiff/population_phl_2018-10-01.tif')#hrsl_phl_v1/hrsl_phl_pop.tif')
    pop_data = rs.zonal_stats(tmp_gdf.geometry, pop_path, stats=["sum"])
    pop_df = pd.DataFrame(pop_data)
    pop_df.columns = ["pop_{}_{}".format(label, i) for i in pop_df.columns]
    # add to main df
    gdf = pd.concat([gdf, ntl_df, pop_df], axis=1)


# save ntl and pop data to csv
df = gdf[[i for i in gdf.columns if i not in ["geometry", "longitude", "latitude"]]]
df_path = os.path.join(data_dir, 'osm-places_ntl_pop_{}.csv'.format(date))
df.to_csv(df_path, index=False, encoding="utf-8")
