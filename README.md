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

5. Setup `config.ini`

6. Run `dhs_clusters.py` to prepare DHS data

7. Run `osm_features.py` to prepare OSM data

8. Run `models.py` to train models and produce figures.
