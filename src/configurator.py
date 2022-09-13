from models import run_models
from configparser import ConfigParser

config = ConfigParser()

config.add_section("main")
config.set("main", "project_dir", "/home/userx/Desktop/accessible-poverty-estimates")

# path to SpatiaLite extension
# build check-in 03786a62cd or later
# see README.md for more detailed instructions
# to find existing path: `whereis mod_spatialite.so`
config.set("main", "spatialite_lib_path", "/usr/local/lib/mod_spatialite")

config.set("main", "mlflow_models_location", "sqlite:///%(project_dir)s/mlflow.db")

config.set("main", "project", "PH_2017_DHS")
config.set("main", "indicator", "Wealth Index")

config.add_section("models")
config.set("models", "ntl", "False")
config.set("models", "sub", "True")
config.set("models", "all", "False")
config.set("models", "loc", "False")
config.set("models", "all-osm", "False")
config.set("models", "sub-osm", "False")
config.set("models", "all-geo", "False")
config.set("models", "sub-geo", "False")
config.set("models", "sub-osm-ntl", "False")
config.set("models", "all-osm-ntl", "False")
config.set("models", "sub-osm-all-geo", "False")

config.add_section("mlflow_tags")
config.set("mlflow_tags", "run_group", "test")
config.set("mlflow_tags", "version", "1.0.0")

config.add_section("PH_2017_DHS")
config.set("PH_2017_DHS", "output_name", "PH_2017_DHS")
config.set("PH_2017_DHS", "country_name", "philippines")
config.set("PH_2017_DHS", "osm_date", "210101")
config.set("PH_2017_DHS", "dhs_hh_file_name", "PHHR71FL")
config.set("PH_2017_DHS", "dhs_geo_file_name", "PHGE71FL")
config.set("PH_2017_DHS", "country_utm_epsg_code", "32651")
config.set("PH_2017_DHS", "geom_id", "DHSID")
config.set("PH_2017_DHS", "geom_label", "dhs-buffers")
config.set("PH_2017_DHS", "geoquery_data_file_name","merge_phl_dhs_buffer")
config.set("PH_2017_DHS", "ntl_year", "2016")
config.set("PH_2017_DHS", "geospatial_variable_years", "[2016]")
config.set("PH_2017_DHS", "gb_iso3", "PHL")

config.add_section("PH_2017_DHS.tags")
config.set("PH_2017_DHS.tags", "example_tag", "hello")

run_models(config)
