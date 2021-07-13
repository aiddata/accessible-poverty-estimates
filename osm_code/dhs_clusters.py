"""
python 3.8

portions of code and/or methodology based on https://github.com/thinkingmachines/ph-poverty-mapping


Prepare DHS, VIIRS, HRSL data

# VIIRS NTL

download viirs nighttime lights
 - 2016, average masked

$ project_dir="/home/userv/Desktop/PHL_WORK"
$ mkdir -p ${project_dir}/data/{viirs,dhs}

$ cd ${project_dir}/data/viirs
$ wget https://eogdata.mines.edu/nighttime_light/annual/v20/2016/VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif.gz
$ gunzip VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif.gz
$ gdalinfo VNL_v2_npp_2016_global_vcmslcfg_c202102150000.average_masked.tif.gz

(optional) use QGIS to clip VIIRS NTL layer to Philippines extent
    - this will drastically reduce file size
    - extent does not need to be exact, just make sure it includes all of Philippines
    - output path: "<project_dir>/data/viirs/clipped_VNL_v2_npp_2016_global_vcmslcfg_c202102150000_average_masked.tif"


# HRSL POPULATION

$ cd ${project_dir}/data
$ wget https://www.ciesin.columbia.edu/repository/hrsl/hrsl_phl_v1.zip
$ unzip hrsl_phl_v1.zip

# DHS CLUSTERS AND OUTCOMES

download DHS data (requires registration) into ${project_dir}/data/dhs
https://dhsprogram.com/data/dataset/Philippines_Standard-DHS_2017.cfm?flag=1
2017 household recode and gps (PHHR71DT.ZIP, PHGE71FL.ZIP)
unzip both files
"""

import os
import pandas as pd
import geopandas as gpd
import rasterstats as rs
import pyreadstat


project_dir = "/Users/sasanfaraj/Desktop/folders/AidData/PHL_WORK" #"/home/userw/Desktop/PHL_WORK"
data_dir = os.path.join(project_dir, 'data')


# ---------------------------------------------------------
# prepare DHS cluster indicators

dhs_file = os.path.join(data_dir, 'dhs/PHHR71DT/PHHR71FL.DTA') #'dhs/PH_2017_DHS_03022021_2055_109036/PHHR71DT/PHHR71FL.DTA')
dhs_dict_file = os.path.join(data_dir,  'dhs/PHHR71DT/PHHR71FL.DO') #'PH_2017_DHS_03022021_2055_109036/PHHR71DT/PHHR71FL.DO')

dhs = pd.read_stata(dhs_file, convert_categoricals=False)

"""
Column      - Description
hv001       - Cluster number
hv271       - Wealth index factor score combined (5 decimals)
hv108_01    - Education completed in single years
hv206       - Has electricity
hv204       - Access to water (minutes)
hv226       - Type of cooking fuel 
hv208       - Has television
"""            #hv226 and hv 206 add by sasan

hh_data = dhs[["hv001", "hv271", "hv108_01", "hv206", "hv204"]] #, "hv208", "hv226"

# 996 = water is on property
hh_data["hv204"] = hh_data["hv204"].replace(996, 0)

cluster_data = hh_data.groupby("hv001").agg({
    "hv271": "mean",
    "hv108_01": "mean",
    "hv206": "mean",
    "hv204": "median"
   # "hv208": "mean",
    #"hv226": "mean"
}).reset_index().dropna(axis=1)

print('Cluster Data Dimensions: {}'.format(cluster_data.shape))

# cluster_data.columns = ["cluster_id", "wealthscore", "education_years", "electricity", "time_to_water"]

cluster_data.columns = [
    'Cluster number',
    'Wealth Index',
    'Education completed (years)',
    'Access to electricity',
    'Access to water (minutes)'
    #'Has television',
    #'Type of cooking feul'
]

# save aggregated cluster indicators
dhs_out_path = os.path.join(data_dir, "dhs_indicators.csv")
cluster_data.to_csv(dhs_out_path, encoding="utf-8")
# # ---------------------------------------------------------
#read in IR file along with the label file
# dhs_ir_file = os.path.join(data_dir, 'dhs/PHIR71DT/PHIR71FL.DTA')
# dhs_ir_dict_file = os.path.join(data_dir,  'dhs/PHHR71DT/PHIR71FL.DO')

# """

# v001        - Cluster number 
# v171b       - Internet usage in the last month 

# """

# dhs_ir = pd.read_stata(dhs_ir_file, convert_categoricals=False)
# print(dhs_ir.shape)


# ir_data = dhs_ir[["v001","v171b"]]

# ir_data = ir_data.groupby("v001").agg({
#     "v171b": "mean",
# }).reset_index().dropna(axis=1)

# ir_data.columns = [
#     'Cluster number',
#     'Internet usage in the last month'
# ]


# ir_out_path = os.path.join(data_dir, "ir_indicators.csv")
# ir_data.to_csv(ir_out_path, encoding="utf-8")


# # ---------------------------------------------------------
 #prepare DHS cluster geometries

shp_path = os.path.join(data_dir, 'dhs/PHGE71FL/PHGE71FL.shp')#PH_2017_DHS_03022021_2055_109036/PHGE71FL/PHGE71FL.shp')
raw_gdf = gpd.read_file(shp_path)

# drop locations without coordinates
raw_gdf = raw_gdf.loc[(raw_gdf["LONGNUM"] != 0) & (raw_gdf["LATNUM"] != 0)].reset_index(drop=True)

raw_gdf.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"}, inplace=True)

def buffer(geom, urban_rural):
    """Create DHS cluster buffers

    Buffer size:
        - 2km for urban
        - 5km for rural (1% of rural have 10km displacement, but ignoring those)

    Metric units converted to decimal degrees by dividing by width of one decimal
    degree in km at equator. Not an ideal buffer created after reprojecting,
    but good enough for this application.
    """
    if urban_rural == "U":
        return geom.buffer(2000)
    elif urban_rural == "R":
        return geom.buffer(5000)
    else:
        raise ValueError("Invalid urban/rural identified ({})".format(urban_rural))


gdf = raw_gdf.copy(deep=True)
# convert to UTM 51N first (meters) then back to WGS84 (degrees)
gdf = gdf.to_crs("EPSG:32651") # UTM 51N
# generate buffer
gdf.geometry = gdf.apply(lambda x: buffer(x.geometry, x.URBAN_RURA), axis=1)
gdf = gdf.to_crs("EPSG:4326") # WGS84

# save buffer geometry as geojson
geo_path = os.path.join(data_dir, 'dhs_buffers.geojson')
gdf.to_file(geo_path, driver='GeoJSON')


#----------------------------------------------------------
# prepare additional geospatial data

# get ntl zonal stats
ntl_path = os.path.join(data_dir, 'viirs/VNL_v2_npp_2016_global_vcmslcfg_c202101211500.average.tif') #clipped_VNL_v2_npp_2016_global_vcmslcfg_c202102150000_average_masked.tif')
ntl_data = rs.zonal_stats(gdf.geometry, ntl_path, stats=["mean", "min", "max", "median", "sum"])
ntl_df = pd.DataFrame(ntl_data)
ntl_df.columns = ["ntl_{}".format(i) for i in ntl_df.columns]
gdf_merge = pd.concat([gdf, ntl_df], axis=1)

# get pop zonal stats
pop_path = os.path.join(data_dir,'population_phl_2018-10-01_geotiff/population_phl_2018-10-01.tif') #'hrsl_phl_v1/hrsl_phl_pop.tif')
pop_data = rs.zonal_stats(gdf.geometry, pop_path, stats=["sum"])
pop_df = pd.DataFrame(pop_data)
pop_df.columns = ["pop_{}".format(i) for i in pop_df.columns]
gdf_merge = pd.concat([gdf_merge, pop_df], axis=1)

# join DHS spatial covars
# covars_path = os.path.join(data_dir, 'dhs/PH_2017_DHS_03022021_2055_109036/PHGC72FL/PHGC72FL.csv')
# covar_data = pd.read_csv(covars_path)
# gdf_merge = gdf_merge.merge(covar_data, on="DHSID", how="left")

gdf_merge["DHSCLUST"] = gdf_merge["DHSCLUST"].astype(int)


# ---------------------------------------------------------
# prepare combined dhs cluster data

# merge geospatial data with dhs hr indicators
gdf_merge = gdf_merge.merge(cluster_data, left_on="DHSCLUST", right_on="Cluster number", how="inner")

#merge geospatial data with dhs ir indicators

# gdf_merge = gdf_merge.merge(ir_data, left_on="DHSCLUST", right_on="Cluster number", how="inner")
# gdf_merge.drop("Cluster number_y", axis=1)
# gdf_merge.rename(columns={'Cluster number_x':'Cluster number'},inplace=True)


# output all dhs cluster data to CSV
final_df = gdf_merge[[i for i in gdf_merge.columns if i != "geometry"]]
final_path =  os.path.join(data_dir, 'dhs_data.csv')
final_df.to_csv(final_path, index=False, encoding='utf-8')
