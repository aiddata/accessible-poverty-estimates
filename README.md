Using random forests regressions (RFRs) and features derived from OpenStreetMap (OSM) and other geospatial data to produce estimates of development indicators such as poverty.



Builds on work published by [Thinking Machines Data Science](https://github.com/thinkingmachines/ph-poverty-mapping).



## Instructions

1. Clone/download repo

2. Create Conda environment

	- To create a new environment:
        - From scratch
        ```
        conda create -n osm-rf python=3.9 -c conda-forge
        conda activate osm-rf
        conda install -c conda-forge --file requirements.txt
        ```
        - From env file
		```
		conda env create -f environment.yml
        conda activate osm-rf
		```
	- To update your environment (if needed):
		```
		conda env update --prefix ./env --file environment.yml  --prune
	- To export/save your environment (if needed):
		```
		conda env export > environment.yml
		```

3. Download DHS data
    - Register a DHS account, login, and then create a project to request access to the survey data for your countries of interest
        - Be sure to request access to both the Standard DHS survey data and the GIS/GPS data for all countries you wish to download data for.
    - Once your access to both the Standard DHS and GIS/GPS data is approved (may be approved at different times), use either your [account project page](https://dhsprogram.com/data/dataset_admin/index.cfm) to access the download manager.
    - Using the download manager, select the Household Recode Stata dataset (.dta) and the Geographic Data Shape file (.shp) then click the "Process Selected Files for Download" button
    - Unzip the download in the `data/dhs` directory of this repo

4. Download OSM data
    - Navigate to country page on [Geofabrik](https://download.geofabrik.de)
        - e.g., https://download.geofabrik.de/asia/philippines.html
    - Click the "raw directory index" link
    - Download the OSM data (shp.zip file) with your desired timestamp (e.g., "philippines-210101-free.shp.zip")
    - Unzip the download in the `data/osm` directory of this repo
    - Note: If you are considering using older snapshots of the OSM database to align with DHS round years, historical coverage of OSM may be limited and impede model performance. For most developing countries, the amount of true historical coverage lacking in older snapshots is likely to be far greater than the amount of actual building/road/etc development since the last DHS round. Neither is ideal, but we expect using more recent OSM data will be more realistic overall. Please consider your particular use case and use your judgement to determine what is best for your application.

5. Setup `config.ini`
    - Note: If you are replicating examples from this repo, you only need to modify the `project_dir` and `project` in the [main] section of the config file
    - `project_dir` - path to the root of the cloned repo
    - `project` - specifies which subsection to use for project specific configurations (e.g., To replicate work using the Philippines 2017 DHS, use "PH_2017_DHS")

    - Project specific configurations (e.g., within the [PH_2017_DHS] section):
        - `country_name` - name of country based on the OSM download file (e.g., "philippines" for "philippines-210101-free.shp.zip")
        - `osm_date` - timestamp from the OSM download (e.g. "210101" for "philippines-210101-free.shp.zip")
        - `dhs_round` - base component of DHS download zip. All downloads will have additional characters you can ignore. (e.g., "PH_2017_DHS" for "PH_2017_DHS_02012021_2025_149015")
        - `dhs_hh_file_name` - filename of the DHS household recode data. You can determine this via the DHS download manager or using your downloaded files. (e.g. "PHHR71DT" for the 2017 Philippines DHS)
        - `dhs_geo_file_name` - filename of the DHS geographic data. You can determine this via the DHS download manager or using your downloaded files. (e.g. "PHGE71FL" for the 2017 Philippines DHS)
        - `country_utm_epsg_code` - EPSG code specific to the country of interest. Typically based on the UTM code for the country (e.g., "32651" to specify UTM 51N for the Philippines). See [example search](https://epsg.io/?q=Philippines+UTM)
            - If your area of interest spans multiple UTM zones you can select the best fit or determine another suitable EPSG code (main purpose of code is to support reprojection of WGS84 data for accurate calculation of the area/length of features in meters).
        - `geom_id` - "DHSID" for use with all DHS data. Could be modified for use with alternative data sources.
        - `geom_label` - "dhs-buffers" based on file name generation currently hard coded. Could be modified for use with alternative data sources.
        - `geoquery_data_file_name` - base file name of CSV file containing geospatial variables from GeoQuery. Could be modified to use alternative data sources (this would also require adjusting variables in `models.py`)
        - `ntl_year` - Base year to use for nighttime lights (and potentially other geospatial variables)
        - `geospatial_variable_years` - List of years to include for time series geospatial variables (limited to what is available in data downloaded from GeoQuery)


6. Run `dhs_clusters.py` to prepare DHS data

7. Run `gen_spatialite.py` to convert OSM buildings/roads shapefiles to SpatiaLite databases

8. Run `osm_features.py` to prepare OSM data
    - Note: If you are adapting this code for another country, be sure to update the OSM crosswalk files before this step. The `crosswalk_gen.py` script can be used to do this.

9. Run `models.py` to train models and produce figures.
