Using random forests regressions (RFRs) and features derived from OpenStreetMap (OSM) and other geospatial data to produce estimates of development indicators such as poverty.



Builds on work published by [Thinking Machines Data Science](https://github.com/thinkingmachines/ph-poverty-mapping).



## Instructions

1. Clone repo

2. Create Conda environment

	- To create a new environment:
        From scratch
        ```
        conda create -n osm-rf python=3.9 -c conda-forge
        conda activate osm-rf
        conda install -c conda-forge --file requirements.txt
        ```
        From env file
		```
		conda env create -f environment.yml
        conda activate osm-rf
		```
    - Activate and install pip
        ```
        pip install mpi4py
        ```
	- To update your environment (if needed):
		```
		conda env update --prefix ./env --file environment.yml  --prune
	- To export your environment (if needed):
		```
		conda env export > environment.yml
		```

3. Download DHS data

4. Download OSM data
    - Note: If you are considering using older snapshots of the OSM database to align with DHS round years, historical coverage of OSM may be limited and impede model performance. For most developing countries, the amount of true historical coverage lacking in older snapshots is likely to be far greater than the amount of actual building/road/etc development since the last DHS round. Neither is ideal, but we expect using more recent OSM data will be more realistic overall. Please consider your particular use case and use your judgement to determine what is best for your application.

5. Setup `config.ini`

6. Run `dhs_clusters.py` to prepare DHS data

7. Run `osm_features.py` to prepare OSM data
    - Note: If you are adapting this code for another country, be sure to update the OSM crosswalk files before this step. The `crosswalk_gen.py` script can be used to do this.

8. Run `models.py` to train models and produce figures.
