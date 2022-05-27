"""

"""

import os
import configparser
import glob
from pathlib import Path

import pandas as pd
import geopandas as gpd


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

dhs_hh_file_name = config[project]['dhs_hh_file_name']

data_dir = project_dir / 'data'



# ---------------------------------------------------------

"""
Column      - Description
hv001       - Cluster number
hv219       - Head of household sex (1 male, 2 female)
"""

dhs_path = Path(glob.glob(str(data_dir / 'dhs' / '**' / f'{dhs_hh_file_name}.DTA' ), recursive=True)[0])

dhs_df = pd.read_stata(dhs_path, convert_categoricals=False)

male_dhs_df = dhs_df.loc[dhs_df['hv219'] == 1].copy()
male_path = dhs_path.parent / f'{dhs_path.stem}_male.DTA'
male_dhs_df.to_stata(male_path, write_index=False)

female_dhs_df = dhs_df.loc[dhs_df['hv219'] == 2].copy()
female_path = dhs_path.parent / f'{dhs_path.stem}_female.DTA'
female_dhs_df.to_stata(female_path, write_index=False)

# weighted_dhs_df = dhs_df.loc[dhs_df['hv219'] == ???]
# weighted_path = dhs_path.parent / f'{dhs_path.stem}_weighted.DTA'
# weighted_dhs_df.to_stata(weighted_path, write_index=False)

print(dhs_df['hv219'].value_counts())
