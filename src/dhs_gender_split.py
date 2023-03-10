"""

"""
import sys
import os
import glob
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

import pandas as pd
import geopandas as gpd

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "config.ini"

if config_file not in os.listdir():
    raise FileNotFoundError(
        f"{config_file} file not found. Make sure you run this from the root directory of the repo and file exists."
    )

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

dhs_hh_file_name = config[project]['dhs_hh_file_name']

data_dir = project_dir / 'data'


# ---------------------------------------------------------


dhs_path = Path(glob.glob(str(data_dir / 'dhs' / '**' / f'{dhs_hh_file_name}.DTA' ), recursive=True)[0])
dhs_df = pd.read_stata(dhs_path, convert_categoricals=False)

full_dhs_path = dhs_path.parent / f"{dhs_path.stem}_all_hh.csv"
dhs_df.to_csv(full_dhs_path, index=False)

full_cluster_df = dhs_df.groupby('hv001').agg({'hv271': 'mean'}).reset_index()
full_cluster_df.columns = ['cluster_id', 'wealth_index']
full_cluster_path = dhs_path.parent / f"{dhs_path.stem}_all_cluster.csv"
full_cluster_df.to_csv(full_cluster_path, index=False)




class GenPaths():
    def __init__(self, parent, stem):
        self.parent = str(parent)
        self.stem = str(stem)
        self.template = self.parent + '/' + self.stem + '_{id}_{gender}_{level}.{ext}'
    def export(self, df, id, gender):
        stata_path = self.template.format(id=id, gender=gender, level='hh', ext='DTA')
        csv_path = self.template.format(id=id, gender=gender, level='hh', ext='csv')
        df.to_stata(stata_path, write_index=False)
        df.to_csv(csv_path, index=False)
        cluster_df = self.agg_to_cluster(df)
        cluster_path = self.template.format(id=id, gender=gender, level='cluster', ext='csv')
        cluster_df.to_csv(cluster_path, index=False)
        # print(f"{len(df)} households aggregated to {len(cluster_df)} clusters")
    def agg_to_cluster(self, df):
        # agg data to clusters
        # provide stats on gendered subsets by household and cluster level?
        cluster_df = (
            df.groupby("hv001")
            .agg({'hv271': 'mean'})
            # .drop(columns="hv001")
            .reset_index()
            # .dropna(axis=1)
        )
        cluster_df.columns = ['cluster_id', 'wealth_index']
        return cluster_df


GP = GenPaths(dhs_path.parent, dhs_path.stem)


# ---------------------------------------------------------

# split by head of household
#   - classify household according to gender of head of household
#   - identifier: hoh (short for: Head Of Household)
#   - question hv219 - Head of household sex (1 male, 2 female)

print(dhs_df['hv219'].value_counts())

hoh_male_df = dhs_df.loc[dhs_df['hv219'] == 1].copy()
hoh_female_df = dhs_df.loc[dhs_df['hv219'] == 2].copy()

GP.export(hoh_male_df, 'hoh', 'male')
GP.export(hoh_female_df, 'hoh', 'female')


# hoh_weighted_df = dhs_df.loc[dhs_df['hv219'] == ???]
# GP.export(hoh_weighted_df, 'hoh', 'weighted')


# ---------------------------------------------------------


# split by more male than female in household?
#   - if any adult male in household, classify male household
#   - identifier == "anym" (short for: ANY Male)
#   - question: hv011 - Number of men in household eligible for men's survey (typically means age 15-49)


print(dhs_df['hv011'].value_counts())

anym_male_df = dhs_df.loc[dhs_df['hv011'] > 0].copy()
anym_female_df = dhs_df.loc[dhs_df['hv011'] == 0].copy()

GP.export(anym_male_df, 'anym', 'male')
GP.export(anym_female_df, 'anym', 'female')



# ---------------------------------------------------------
# split based on assets

'''
male

Bicycle
x["hv210"]

Motorcycle/scooter
x["hv211"]

Bank account
x["hv247"]

Owns land suitable for agriculture
x["hv244"]


female
Type of toilet facility: ventilated improved pit latrine (private or shared)
x["hv205"] == 21

Source of drinking water: public tap/standpipe
x["hv201"] == 13

Type of cooking fuel: charcoal
x["hv226"] == 7
'''

# split by male assets

def has_male_assets(x):
    return x['hv210'] > 0 or x["hv211"] > 0 or x["hv247"] > 0 or x["hv244"] > 0

massets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()
massets_female_df = dhs_df.loc[~dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()

GP.export(massets_male_df, 'massets', 'male')
GP.export(massets_female_df, 'massets', 'female')


# split by female assets

def has_female_assets(x):
    return x['hv205'] == 21 or x["hv201"] == 13 or x["hv226"] == 7

fassets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()
fassets_female_df = dhs_df.loc[~dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GP.export(fassets_male_df, 'fassets', 'male')
GP.export(fassets_female_df, 'fassets', 'female')


# use subset of postiively male/female identified assets only
#   - drop overlap?

mfassets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()
mfassets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GP.export(mfassets_male_df, 'mfassets', 'male')
GP.export(mfassets_female_df, 'mfassets', 'female')
