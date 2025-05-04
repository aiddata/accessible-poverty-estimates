"""

"""
import sys
import os
import glob
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
import shutil

import numpy as np
import pandas as pd

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "eqai_config.ini"

if config_file not in os.listdir():
    raise FileNotFoundError(
        f"{config_file} file not found. Make sure you run this from the root directory of the repo and file exists."
    )

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file)

project_dir = Path(config["main"]["project_dir"])
data_dir = project_dir / 'data'


dhs_name = "GH_2014_DHS"
dhs_hh_file_name = "GHHR72FL"


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
if not dry_run:
    dhs_df.to_csv(full_dhs_path, index=False)

full_cluster_df = dhs_df.groupby('hv001').agg({'hv271': 'mean'}).reset_index()
full_cluster_df.columns = ['cluster_id', 'wealth_index']
full_cluster_path = dhs_path.parent / f"{dhs_path.stem}_all_cluster.csv"
if not dry_run:
    full_cluster_df.to_csv(full_cluster_path, index=False)



class GenderClassData():

    def __init__(self, id, dhs_path, dhs_name, dry_run=False):
        self.id = id
        self.dhs_path = dhs_path
        self.parent = str(dhs_path.parent)
        self.stem = str(dhs_path.stem)
        self.template = self.parent + '/' + self.stem + '_{id}_{gender}_{level}.{ext}'
        self.all_df = pd.DataFrame()
        self.all_cluster = pd.DataFrame()
        self.male_df = pd.DataFrame()
        self.male_cluster = pd.DataFrame()
        self.female_df = pd.DataFrame()
        self.female_cluster = pd.DataFrame()
        self.dry_run = dry_run

        self.dhs_name = dhs_name
        self.src_dir = dhs_path.parent.parent.parent / "outputs" / dhs_name
        self.src_df = pd.read_csv(self.src_dir / "final_data.csv")
        self.src_df.drop(columns=["Wealth Index"], inplace=True)

        self.base_dst_dir = dhs_path.parent.parent.parent / "outputs"

    def all_exist(self):
        return len(self.male_df) > 0 and len(self.female_df) > 0 and len(self.male_cluster) > 0 and len(self.female_cluster) > 0

    def set_male(self, df):
        self.male_df = df

    def set_female(self, df):
        self.female_df = df

    def set_all(self, df):
        self.all_df = df

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

    def cluster(self, genders=["male", "female"]):
        if "male" in genders:
            self.male_cluster = self.agg_to_cluster(self.male_df)
        if "female" in genders:
            self.female_cluster = self.agg_to_cluster(self.female_df)
        if "all" in genders:
            self.all_cluster = self.agg_to_cluster(self.all_df)

    def describe(self, verify_exists=True):
        if verify_exists and not self.all_exist():
            raise Exception('Male and Female dataframes for household and cluster not set')

        description = f"""
        Classification Strategy: {self.id}

        {len(self.male_df)} male households aggregated to {len(self.male_cluster)} clusters
        {len(self.female_df)} female households aggregated to {len(self.female_cluster)} clusters
        {len(self.all_df)} all households aggregated to {len(self.all_cluster)} clusters

        """
        print(description)

    def export(self, genders=["male", "female"], verify_exists=True):
        if verify_exists and not self.all_exist():
            raise Exception('Male and Female dataframes not set')

        if self.dry_run:
            print('Dry run, not exporting')

        for g in genders:
            self._export(g)
            self._integrate(g)


    def _export(self, gender):
        if gender == "male":
            df = self.male_df
            cluster = self.male_cluster
        elif gender == "female":
            df = self.female_df
            cluster = self.female_cluster
        elif gender == "all":
            df = self.all_df
            cluster = self.all_cluster
        else:
            raise ValueError(f"Invalid gender value provided:{gender}")

        stata_path = self.template.format(id=self.id, gender=gender, level='hh', ext='DTA')
        csv_path = self.template.format(id=self.id, gender=gender, level='hh', ext='csv')
        if not self.dry_run:
            df.to_stata(stata_path, write_index=False)
            df.to_csv(csv_path, index=False)

        cluster_path = self.template.format(id=self.id, gender=gender, level='cluster', ext='csv')
        if not self.dry_run:
            cluster.to_csv(cluster_path, index=False)


    def _integrate(self, gender):
        dst_name = f"{self.dhs_name}_{self.id}_{gender}"
        dst_dir = self.base_dst_dir/ dst_name
        if not self.dry_run and not dst_dir.exists():
            shutil.copytree(self.src_dir, dst_dir)

        classification_name = dst_name.split("_")[-2]
        classification_gender = dst_name.split("_")[-1]
        gender_cluster_path = self.dhs_path.parent / f"GHHR72FL_{classification_name}_{classification_gender}_cluster.csv"

        try:
            gender_cluster_df = pd.read_csv(gender_cluster_path)
        except FileNotFoundError:
            if self.dry_run:
                print("You are doing a dry run. The resulting files may not exist to integrate if export has not been run before.")
            raise

        gender_cluster_df["DHSID"] =  gender_cluster_df.cluster_id.apply(lambda x: f'GH201400000{str(x).zfill(3)}')
        gender_cluster_df.rename({"wealth_index": "Wealth Index"}, axis=1, inplace=True)
        gender_cluster_df.drop(columns=["cluster_id"], inplace=True)

        new_df = self.src_df.merge(gender_cluster_df, on="DHSID", how="inner")

        output_path = dst_dir / "final_data.csv"
        if not self.dry_run:
            new_df.to_csv(output_path, index=False)


# ---------------------------------------------------------

run_num_str = 5

suffix_str = f"-alt-{str(run_num_str).zfill(2)}"

# suffix_str = ""

# ---------------------------------------------------------

small_cluster_df_list = []

for cid in dhs_df.hv001.unique():
    cdf = dhs_df.loc[dhs_df.hv001 == cid].copy()
    small_cluster_df = dhs_df.sample(9)
    small_cluster_df_list.append(small_cluster_df)

small_df = pd.concat(small_cluster_df_list)


GCD0small = GenderClassData(f'small{suffix_str}', dhs_path, dhs_name, dry_run=dry_run)
GCD0small.set_all(small_df)
GCD0small.cluster(genders=["all"])
GCD0small.describe(verify_exists=False)
GCD0small.export(genders=["all"], verify_exists=False)


medium_cluster_df_list = []

for cid in dhs_df.hv001.unique():
    cdf = dhs_df.loc[dhs_df.hv001 == cid].copy()
    medium_cluster_df = dhs_df.sample(19)
    medium_cluster_df_list.append(medium_cluster_df)

medium_df = pd.concat(medium_cluster_df_list)


GCD0medium = GenderClassData(f'medium{suffix_str}', dhs_path, dhs_name, dry_run=dry_run)
GCD0medium.set_all(medium_df)
GCD0medium.cluster(genders=["all"])
GCD0medium.describe(verify_exists=False)
GCD0medium.export(genders=["all"], verify_exists=False)

# ---------------------------------------------------------


# split by head of household with each cluster adjusted so that
# male and female counts per cluster are equal
#   - adjustment is done per cluster and involves randomly dropping
#     households from the gender with more households in that cluster

hoheq_cluster_counts = {}
hoheq_male_cluster_df_list = []
hoheq_female_cluster_df_list = []
for cid in dhs_df.hv001.unique():
    cdf = dhs_df.loc[dhs_df.hv001 == cid].copy()
    male_cluster_df = cdf.loc[cdf['hv219'] == 1]
    female_cluster_df = cdf.loc[cdf['hv219'] == 2]
    cluster_male_count = len(male_cluster_df)
    cluster_female_count = len(female_cluster_df)
    min_sample_size = min(cluster_male_count, cluster_female_count)
    max_sample_size = max(cluster_male_count, cluster_female_count)
    hoheq_cluster_counts[cid] = {
        "min": min_sample_size,
        "max": max_sample_size,
        "total": min_sample_size + max_sample_size
    }
    eq_male_cluster_df = male_cluster_df.sample(min_sample_size)
    eq_female_cluster_df = female_cluster_df.sample(min_sample_size)
    hoheq_male_cluster_df_list.append(eq_male_cluster_df)
    hoheq_female_cluster_df_list.append(eq_female_cluster_df)

print(f"""
Median count of cluster households for minority gender: {np.median([i['min'] for i in hoheq_cluster_counts.values()])}
Median count of cluster households for majority gender: {np.median([i['max'] for i in hoheq_cluster_counts.values()])}
Median total count of cluster households: {np.median([i['total'] for i in hoheq_cluster_counts.values()])}
""")

hoheq_male_df = pd.concat(hoheq_male_cluster_df_list)
hoheq_female_df = pd.concat(hoheq_female_cluster_df_list)

GCD1eq = GenderClassData(f'hoheq{suffix_str}', dhs_path, dhs_name, dry_run=dry_run)
GCD1eq.set_male(hoheq_male_df)
GCD1eq.set_female(hoheq_female_df)
GCD1eq.cluster()
GCD1eq.describe()
GCD1eq.export()

# ---------------------------------------------------------

exit()

# ---------------------------------------------------------

# split by head of household
#   - classify household according to gender of head of household
#   - identifier: hoh (short for: Head Of Household)
#   - question hv219 - Head of household sex (1 male, 2 female)

print(dhs_df['hv219'].value_counts())

hoh_male_df = dhs_df.loc[dhs_df['hv219'] == 1].copy()
hoh_female_df = dhs_df.loc[dhs_df['hv219'] == 2].copy()
# hoh_weighted_df = dhs_df.loc[dhs_df['hv219'] == ???]

GCD1 = GenderClassData('hoh', dhs_path, dhs_name, dry_run=dry_run)
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

GCD2 = GenderClassData('anym', dhs_path, dhs_name, dry_run=dry_run)
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

GCD3 = GenderClassData('massets', dhs_path, dhs_name, dry_run=dry_run)
GCD3.set_male(massets_male_df)
GCD3.set_female(massets_female_df)
GCD3.cluster()
GCD3.describe()
GCD3.export()


# split by female assets

fassets_male_df = dhs_df.loc[~dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()
fassets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GCD4 = GenderClassData('fassets', dhs_path, dhs_name, dry_run=dry_run)
GCD4.set_male(fassets_male_df)
GCD4.set_female(fassets_female_df)
GCD4.cluster()
GCD4.describe()
GCD4.export()


# use subset of postiively male/female identified assets only
#   - retain overlap

mf1assets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()
mf1assets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()

GCD5 = GenderClassData('mf1assets', dhs_path, dhs_name, dry_run=dry_run)
GCD5.set_male(mf1assets_male_df)
GCD5.set_female(mf1assets_female_df)
GCD5.cluster()
GCD5.describe()
GCD5.export()

# use subset of postiively male/female identified assets only
#   - drop overlap

mf2assets_male_df = dhs_df.loc[dhs_df.apply(lambda x: has_male_assets(x), axis=1) & ~dhs_df.apply(lambda x: has_female_assets(x), axis=1)].copy()
mf2assets_female_df = dhs_df.loc[dhs_df.apply(lambda x: has_female_assets(x), axis=1) & ~dhs_df.apply(lambda x: has_male_assets(x), axis=1)].copy()

GCD6 = GenderClassData('mf2assets', dhs_path, dhs_name, dry_run=dry_run)
GCD6.set_male(mf2assets_male_df)
GCD6.set_female(mf2assets_female_df)
GCD6.cluster()
GCD6.describe()
GCD6.export()


# ---------------------------------------------------------
# ---------------------------------------------------------



GCD1.describe()
GCD1eq.describe()
GCD2.describe()
GCD3.describe()
GCD4.describe()
GCD5.describe()
GCD6.describe()
