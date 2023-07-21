# Accessible Poverty Estimates

We utilize random forests regressions (RFRs) and features derived from a range of geospatial datasets (nighttime lights, landcover, OpenStreetMap features, and more) to produce estimates of development indicators such as household wealth/poverty. The core random forest code builds on work published by [Thinking Machines Data Science](https://github.com/thinkingmachines/ph-poverty-mapping), and has been extended to streamline model development across use cases. While our examples focus on the use of Demographic and Health Survey (DHS) data for wealth metrics used to train models, it can be extended to work with other sources of data as well.

This repository also supports [AidData's work funded by USAID's Equitable AI Challenge](https://www.aiddata.org/blog/aiddata-wins-award-from-usaids-equitable-ai-challenge-for-project-on-gender-bias-in-ai-created-poverty-measures) to evaluate gender influence in machine learning based poverty estimation approaches. AidData's work explored the 2014 Ghana DHS round by comparing models trained on male or female households. Code and steps specific to replicating or extending work from the project are included and detailed in this readme. In addition, core outputs are provided to enable replication and review of the final results of the analysis. Outputs include the MLFlow database containing all model results, joblib files containing trained models, and a range of figures and other supporting products that are included or referenced in the final report. To learn more about this work, visit AidData's page on this project (forthcoming).


## Before You Start

This project requires downloaded data from [The DHS Program](https://dhsprogram.com/). This data is freely available, but you [must register](https://www.dhsprogram.com/data/Registration-Rationale.cfm) for an account. Please see [downloading data](#downloading-data) below for more information.


## Instructions

### Setting up your environment

1. Clone or download this repository
   ```
   git clone https://github.com/aiddata/accessible-poverty-estimates.git
   cd accessible-poverty-estimates
   ```

2. Create Conda environment
  - Create new environment from env file:
    ```
    conda env create --file environment.yaml
    conda activate ape-env
    ```
  - or from scratch (if you prefer):
    ```
    conda create -n ape-env python=3.9 -c conda-forge
    conda activate ape-env
    conda install -c conda-forge --file requirements.txt
    ```
  - to update your environment (if needed):
    ```
    conda activate ape-env
    conda env update --file environment.yaml  --prune
    ```
  - to export/save your environment (if needed):
    ```
    conda env export > environment.yaml
    ```
  - Add the `src` directory to your Conda environment Python path:
    `conda develop /path/to/project/directory/src` (Note: /path/to/project/directory should be the same as your `project_dir` variable in the config.ini)

3. Install [QGIS](https://www.qgis.org/en/site/)
  - Instructions for doing so on your operating system can be found [here](https://www.qgis.org/en/site/forusers/download.html).
  *While this project does not use QGIS directly, it shares the dependency libspatialite. See [docs/libspatialite_install.md](docs/install_libspatialite.md) if you'd prefer to install it manually.*

### Downloading data

1. Download DHS data
    - **Optional:** To explore what data (country/year) are available, run `dhs_availability.py`
        - Note: This script is meant to serve as an aid and may not always work. If you are interested in extending outside of Phase 7 surveys between 2013 and 2020 you may need to adjust based on desired year range or phase of surveys (variables in script) and test / add exceptions to catch differences in naming / country ISO conventions used by DHS vs other sources.
        - The script will also attempt to generate (and print out) text for each DHS survey that can be added directly to the config.ini file (if not included in it - many 2013:2020 surveys already are).
        - Modify `config.ini` as necessary based on DHS data you wish to use
    - [Register a DHS account](https://dhsprogram.com/data/new-user-registration.cfm), login, and then create a project to [request access](https://dhsprogram.com/data/Access-Instructions.cfm) to the survey data for your countries of interest
        - **Important:** Be sure to request access to both the Standard DHS survey data and the GIS/GPS data for all countries you wish to download data for.
    - Once your access to both the Standard DHS and GIS/GPS data is approved (may be approved at different times), use either your [account project page](https://dhsprogram.com/data/dataset_admin/index.cfm) to access the download manager.
    - You can use the download manager to select data for an individual survey or in bulk.
        - Select the Household Recode Stata dataset (.dta) and the Geographic Data Shape file (.shp) for desired countries/surveys
        - Download into the `data/dhs` directory of this repo
    - Once downloaded:
        - Run `extract_dhs_data.py` to automatically unzip the downloaded data.

2. Download OSM data [**skip for replication only**]
    - **Recommended**: For automated/bulk downloads, you can use the `download_osm.py` script to download and convert OSM shapefiles to SpatiaLite
        - First edit the country list in `download_osm.py` to:
            - 1) Include countries you wish to download data for in their relevant regions.
                - See [Geofabrik](https://download.geofabrik.de) for country name spelling and country-region associatations.
            - 2) Use the desired date of archived OSM data
    - **Alternative**: To manually download data for an individual country:
        - Navigate to country page on [Geofabrik](https://download.geofabrik.de)
            - e.g., https://download.geofabrik.de/asia/ghana.html
        - Click the "raw directory index" link
        - Download the OSM data (shp.zip file) with your desired timestamp (e.g., "ghana-210101-free.shp.zip")
        - Unzip the download in the `data/osm` directory of this repo
        - If using the default `osm_features.py` script in a later step, use `gen_spatialite.py` to convert OSM buildings/roads shapefiles to SpatiaLite databases
    - If you are considering using older snapshots of the OSM database to align with DHS round years, historical coverage of OSM may be limited and impede model performance. For most developing countries, the amount of true historical coverage lacking in older snapshots is likely to be far greater than the amount of actual building/road/etc development since the last DHS round. Neither is ideal, but we expect using more recent OSM data will be more realistic overall. Please consider your particular use case and use your judgement to determine what is best for your application.
    - Year/month availability of older OSM data may not always be consistent across countries. For the most part, each year since 2020 will have at least one archived version (e.g., 20200101, 20210101)

3. Access geospatial variable data from GeoQuery [**skip for replication only**]
    - All data provided by GeoQuery is from publicly available sources. Data is freely and publicly accessible directly via GeoQuery for a broad range of administrative and other boundaries. Data for other boundaries can be requested from GeoQuery and a custom extract will be run for free. See https://www.aiddata.org/geoquery/custom-requests for details.
      - The geospatial data associated with the DHS clusters used in this repository were produced using a custom extract from GeoQuery, and the resulting data is included in the repository for replication.


### Setting configuration options

This section describes options that should be set in `config.ini`

**Important:** You must update the `project_dir` and `spatialite_lib_path` in the [main] section of the config file to reflect your computer's paths.

- `project_dir` - path to the root of this repository on your computer
- `spatialite_lib_path` - path to where the SpatiaLite library was installed. Typically will not need to be modified if you're running an Ubuntu-based distribution of Linux.
  - The command `whereis mod_spatialite.so` can help you find it.
    It is likely in /usr/local/lib/

- `prefect_cloud_enabled` - Set to False if not using Prefect cloud
- `prefect_project_name` - Set Prefect project name if using Prefect cloud
- `dask_enabled` - Set to True if using a Dask cluster
- `use_hpc` - Set to True if running on W&M's HPC
- `use_dask_address` - Set to True if using a remote Dask cluster
- `dask_address` - Address of remote Dask cluster
- `projects_to_run` - Specify a comma separate list of projects to run (project configurations and naming discussed below). For example, to replicate work using the Ghana 2014 DHS, use "GH_2014_DHS".
- `run_sub_projects` - Set to True if projects consisting of multiple sub projects should recursively traverse and run models for each sub project included.
- `model_funcs` - Type of models to run based on the input features provided. We suggest leaving this as the default and running all model types. See code for details on which features are included for each model.
- `indicator` - list of DHS indicators to use for modeling. Currently only supports the DHS "Wealth Index"


- Project specific configurations (e.g., within the [GH_2014_DHS] section):
  - `output_name` - name unique to your config section which will be used to determine where output files are saved (e.g., GH_2014_DHS)
  - `country_name` - name of country based on the OSM download file (e.g., "ghana" for "ghana-210101-free.shp.zip")
  - `osm_date` - timestamp from the OSM download (e.g. "210101" for "ghana-210101-free.shp.zip")
  - `dhs_hh_file_name` - filename of the DHS household recode data. You can determine this via the DHS download manager or using your downloaded files. (e.g. "GHHR72FL" for the 2014 Ghana DHS)
  - `dhs_geo_file_name` - filename of the DHS geographic data. You can determine this via the DHS download manager or using your downloaded files. (e.g. "GHGE71FL" for the 2014 Ghana DHS)
  - `country_utm_epsg_code` - EPSG code specific to the country of interest. Typically based on the UTM code for the country (e.g., "32630" to specify UTM 30N for the Ghana). See [example search](https://epsg.io/?q=Ghana+UTM)
    - If your area of interest spans multiple UTM zones you can select the best fit or determine another suitable EPSG code (main purpose of code is to support reprojection of WGS84 data for accurate calculation of the area/length of features in meters).
  - `geom_id` - "DHSID" for use with all DHS data. Could be modified for use with alternative data sources.
  - `geom_label` - "dhs-buffers" based on file name generation currently hard coded. Could be modified for use with alternative data sources.
  - `geoquery_data_file_name` - base file name of CSV file containing geospatial variables from GeoQuery. Could be modified to use alternative data sources (this would also require adjusting variables in `models.py`)
  - `ntl_year` - Base year to use for nighttime lights (and potentially other geospatial variables)
  - `geospatial_variable_years` - List of years to include for time series geospatial variables (limited to what is available in data downloaded from GeoQuery)

MLFlow config settings:

In the [mlflow] section (these should not need to be modified):
- `tracking_uri` - Path to MLFlow databased. Default: sqlite:///${main:project_dir}/mlflow.db
- `registry_uri` - Path to where MLFlow artificacts are recorded. Default: {main:project_dir}/model_registry
- `artifact_location` - Path to where MLFlow artificacts are recorded. Default: ${main:project_dir}/mlruns
- `experiment_name` - Top level experiment name under which all MLFlow records will be placed. Default: accessible-poverty-estimates

In the [mlflow_tags] section:
- `run_group` - label to be recorded with all runs using the current config setting. Useful for tracking specific groups of models being run (e.g., country A vs country B)
- `version` - version number to be recorded with all runs using the current config setting. Useful for refinement when tracking iterations of similar models.


### Using MLflow to track models

[MLflow](https://mlflow.org/) is a platform that helps keep track of machine learning models and their performance. Training models by following the instructions belows will use MLflow to log models to `mlflow.db`, a SQLite database in the top level of this repository.

To access the MLflow dashboard, run the following command:
```
mlflow ui --backend-store-uri=sqlite:///mlflow.db
```
then, navigate to http://localhost:5000 in your web browser.

In the list of runs on the dashboard homepage, click on one to view a parallel coordinates plot and graph of feature importances.

You can also use the mlflow.db provided with the repo to explore the results of AidData's Equitable AI project without running any models yourself. The database contains the results of all models produced across a range of hyperparameters, input features, and other conditions. This mlflow.db can be used along with `scripts/mlflow_plots.py` to replicate figures and analysis from AidData's work in the Equitable AI Challenge.



### Additional setup: Prefect

We utilize [Prefect](https://www.prefect.io/) for workflow orchestration and monitoring jobs for data processing and model training. Prefect can be used without any additional steps, simply leveraging the Python package installed as part of the environment setup. If you wish to leverage more advanced features of Prefect, such as their cloud based monitoring tools, or more refined control over a local/remote Dask cluster to speed up processing, you can learn more about Prefect at [prefect.io](https://www.prefect.io/)


Note: You may need to run the following in your environment's terminal to deal with an issue using Prefect: `prefect config set PREFECT_API_ENABLE_HTTP2=false`


### (Optional) Preparing gender specific subsets

In this section we will cover some specifc steps that can be used to split DHS household data based on gender using various criteria for male or female household classification. If replicating gender specific models, we provide a pregenerated `eqai_config.ini` file that can be used without fully replicating the steps detailed below. Modifying this file slightly will likely be necessary for you to run on your own system (e.g., defining the path to your project directory).

Note: The approaches for separating the DHS data by gender, and subsequently specifying how to modify the config file to use gender specific data, can be adapted for various other applications which involve subsetting an input data source. We will not cover these, but the existing gender specific examples should illustrate how this is possible.

Using the `config.ini`, as setup in the previous section, run `python scripts/dhs_gender_split.py`. This will create gender specific subsets of the DHS dataset specified in `config.ini` based on the gender of the head of household, presence of males in household, and various conditions of asset ownership. See `dhs_gender_split.py` for specifics, or AidData's report on their project as part of USAID's Equitable AI Challenge (forthcoming).

Modify the `eqai_config.ini` to use the same basic settings as the `config.ini` for your project directory, spatialite_lib_path and other environment specific settings you may have modified in the previous section.

In the following sections you will proceed to use the `eqai_config.ini` in place of the `config.ini`.

Note: The default `eqai_config.ini` has multiple lines commented out for the projects_to_run, model_funcs, and version fields set to specifically replicate existing work. Commented and uncommented lines are each preceded by a "Run #" that should be used across all of the three settings if you modify. To fully replicate AidData's existing work, first run using all settings for Run 1, the proceed through Run 5. The outputs based on these settings are required to run the `scripts/mlflow_plots.py` code in order to replicate figures. No additional modification of `scripts/mlflow_plots.py` should be necessary to run and replicate outputs.


### Preparing Data

1. Run `get_dhs_adm_units.py` to identify ADM 1 and 2 units of DHS clusters

2. Run `dhs_clusters.py` to prepare DHS data

3. Run GeoQuery extract using `extract_job.json` produced by `dhs_clusters.py` [**skip for replication only**]
    - Note: if you are extending this code beyond countries with data available in this repo, contact geo@aiddata.org with your `dhs_buffers.geojson` and `extract_job.json` and they will generate the data for you.

4. Run `gen_spatialite.py` to convert OSM buildings/roads shapefiles to SpatiaLite databases [**skip for replication only**]

5. Run `osm_features.py` to prepare OSM data [**skip for replication only**]
    - Note: If you are adapting this code for another country, be sure to update the OSM crosswalk files before this step. The `crosswalk_gen.py` script can be used to do this.
    - The crosswalk files available with the repo have been built from many countries so it is unlikely any major OSM feature types would be missing, but you can use this script to be sure.
    - `crosswalk_gen.py` by default will check every OSM file in your `data/osm` directory. You can change this to check only a single country specified in your `config.ini` by commenting/uncommented lines specified within the script.
    - After running `crosswalk_gen.py` edit the modified CSV files if new feature types were detected. Excel or any other CSV/spreadsheet editor will work for this. Replace any `0` values in the `group` column with a relevant group (see groups already used in crosswalk files for examples such as an OSM building type of "house" being assigned the group of "residential").

6. Run `model_prep.py` to merge all data required for modeling.



### Training Models

1. Run `model_bundle.py` to train models and produce figures.




## License

This work is released under the MIT License. Please see [LICENSE.md](LICENSE.md) for more information.


## Acknowledgements
Underlying random forest based approach for producing poverty estimates builds on work published by [Thinking Machines Data Science](https://github.com/thinkingmachines/ph-poverty-mapping).
