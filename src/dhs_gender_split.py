"""

"""
import sys
import os
import glob
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

import pandas as pd

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "config.ini"

if config_file not in os.listdir():
    raise FileNotFoundError(
        f"{config_file} file not found. Make sure you run this from the root directory of the repo and file exists."
    )

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file)

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

dhs_hh_file_name = config[project]['dhs_hh_file_name']

data_dir = project_dir / 'data'

dry_run = False

# ---------------------------------------------------------


dhs_path = Path(glob.glob(str(data_dir / 'dhs' / '**' / f'{dhs_hh_file_name}.DTA' ), recursive=True)[0])
dhs_raw_df = pd.read_stata(dhs_path, convert_categoricals=False)

# add in data about adult male in households
# TODO: this uses a hardcoded local path to DHS PR file and will not work for other surveys
dhs_pr_path = project_dir / 'equitable-ai/GH_2014_DHS/GHPR72DT/GHPR72FL.DTA'
dhs_pr_raw_df = pd.read_stata(dhs_pr_path, convert_categoricals=False)
dhs_pr_df = dhs_pr_raw_df[['hv001', 'hv002', 'hv102', 'hv104', 'hv105']].copy()
dhs_pr_df['id'] = dhs_pr_df['hv001'].astype(str) +'_'+ dhs_pr_df['hv002'].astype(str)
dhs_pr_df = dhs_pr_df.loc[dhs_pr_df['hv102'] == 1].copy()
dhs_pr_df = dhs_pr_df.loc[(dhs_pr_df['hv105'] > 15) & (dhs_pr_df['hv105'] < 100)].copy()
dhs_pr_df = dhs_pr_df.loc[dhs_pr_df['hv104'] == 1].copy()
dhs_pr_male_df = dhs_pr_df.groupby('id').agg({'hv104': lambda x: 1}).reset_index()
dhs_pr_male_df.rename({'hv104': 'anymale'}, axis=1, inplace=True)

dhs_raw_df['id'] = dhs_raw_df['hv001'].astype(str) +'_'+ dhs_raw_df['hv002'].astype(str)
dhs_df = dhs_raw_df.merge(dhs_pr_male_df, on='id', how='left')
dhs_df['anymale'] = dhs_df['anymale'].fillna(0)


full_dhs_path = dhs_path.parent / f"{dhs_path.stem}_all_hh.csv"
dhs_df.to_csv(full_dhs_path, index=False)

full_cluster_df = dhs_df.groupby('hv001').agg({'hv271': 'mean'}).reset_index()
full_cluster_df.columns = ['cluster_id', 'wealth_index']
full_cluster_path = dhs_path.parent / f"{dhs_path.stem}_all_cluster.csv"
full_cluster_df.to_csv(full_cluster_path, index=False)



class GenderClassData():

    def __init__(self, id, parent, stem, dry_run=False):
        self.id = id
        self.parent = str(parent)
        self.stem = str(stem)
        self.template = self.parent + '/' + self.stem + '_{id}_{gender}_{level}.{ext}'
        self.male_df = pd.DataFrame()
        self.female_df = pd.DataFrame()
        self.male_cluster = pd.DataFrame()
        self.female_cluster = pd.DataFrame()
        self.dry_run = dry_run

    def all_exist(self):
        return len(self.male_df) > 0 and len(self.female_df) > 0 and len(self.male_cluster) > 0 and len(self.female_cluster) > 0

    def set_male(self, df):
        self.male_df = df

    def set_female(self, df):
        self.female_df = df

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

    def cluster(self):
        self.male_cluster = self.agg_to_cluster(self.male_df)
        self.female_cluster = self.agg_to_cluster(self.female_df)

    def describe(self):
        if not self.all_exist():
            raise Exception('Male and Female dataframes for household and cluster not set')

        description = f"""
        Classification Strategy: {self.id}

        {len(self.male_df)} male households aggregated to {len(self.male_cluster)} clusters
        {len(self.female_df)} female households aggregated to {len(self.female_cluster)} clusters

        """
        print(description)

    def export(self):
        if not self.all_exist():
            raise Exception('Male and Female dataframes not set')

        if self.dry_run:
            print('Dry run, not exporting')
            return

        male_stata_path = self.template.format(id=self.id, gender='male', level='hh', ext='DTA')
        male_csv_path = self.template.format(id=self.id, gender='male', level='hh', ext='csv')
        self.male_df.to_stata(male_stata_path, write_index=False)
        self.male_df.to_csv(male_csv_path, index=False)

        female_stata_path = self.template.format(id=self.id, gender='female', level='hh', ext='DTA')
        female_csv_path = self.template.format(id=self.id, gender='female', level='hh', ext='csv')
        self.female_df.to_stata(female_stata_path, write_index=False)
        self.female_df.to_csv(female_csv_path, index=False)

        male_cluster_path = self.template.format(id=self.id, gender='male', level='cluster', ext='csv')
        self.male_cluster.to_csv(male_cluster_path, index=False)

        female_cluster_path = self.template.format(id=self.id, gender='female', level='cluster', ext='csv')
        self.female_cluster.to_csv(female_cluster_path, index=False)


# ---------------------------------------------------------

# split by head of household
#   - classify household according to gender of head of household
#   - identifier: hoh (short for: Head Of Household)
#   - question hv219 - Head of household sex (1 male, 2 female)

print(dhs_df['hv219'].value_counts())

hoh_male_df = dhs_df.loc[dhs_df['hv219'] == 1].copy()
hoh_female_df = dhs_df.loc[dhs_df['hv219'] == 2].copy()
# hoh_weighted_df = dhs_df.loc[dhs_df['hv219'] == ???]

GCD1 = GenderClassData('hoh', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD1.set_male(hoh_male_df)
GCD1.set_female(hoh_female_df)
GCD1.cluster()
GCD1.describe()
GCD1.export()


# ---------------------------------------------------------

# split by any adult male in household
#   - identifier == "anym" (short for: ANY Male)
#   - question: hv011 - Number of men in household eligible for men's survey (typically means age 15-49)


print(dhs_df['anymale'].value_counts())

anym_male_df = dhs_df.loc[dhs_df['anymale'] > 0].copy()
anym_female_df = dhs_df.loc[dhs_df['anymale'] == 0].copy()

GCD2 = GenderClassData('anym', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD2.set_male(anym_male_df)
GCD2.set_female(anym_female_df)
GCD2.cluster()
GCD2.describe()
GCD2.export()


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

def has_male_assets(x):
    return x['hv210'] > 0 or x["hv211"] > 0 or x["hv247"] > 0 or x["hv244"] > 0

def has_female_assets(x):
    return x['hv205'] == 21 or x["hv201"] == 13 or x["hv226"] == 7

# split by male assets

massets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()
massets_female_df = dhs_df.loc[~dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()

GCD3 = GenderClassData('massets', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD3.set_male(massets_male_df)
GCD3.set_female(massets_female_df)
GCD3.cluster()
GCD3.describe()
GCD3.export()


# split by female assets

fassets_male_df = dhs_df.loc[~dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()
fassets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GCD4 = GenderClassData('fassets', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD4.set_male(fassets_male_df)
GCD4.set_female(fassets_female_df)
GCD4.cluster()
GCD4.describe()
GCD4.export()


# use subset of postiively male/female identified assets only
#   - retain overlap

mf1assets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()
mf1assets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GCD5 = GenderClassData('mf1assets', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD5.set_male(mf1assets_male_df)
GCD5.set_female(mf1assets_female_df)
GCD5.cluster()
GCD5.describe()
GCD5.export()

# use subset of postiively male/female identified assets only
#   - drop overlap

mf2assets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1) & ~dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()
mf2assets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1) & ~dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()

GCD6 = GenderClassData('mf2assets', dhs_path.parent, dhs_path.stem, dry_run=dry_run)
GCD6.set_male(mf2assets_male_df)
GCD6.set_female(mf2assets_female_df)
GCD6.cluster()
GCD6.describe()
GCD6.export()


# ---------------------------------------------------------
# ---------------------------------------------------------



GCD1.describe()
GCD2.describe()
GCD3.describe()
GCD4.describe()
GCD5.describe()
GCD6.describe()
