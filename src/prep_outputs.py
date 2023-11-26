# this file should no longer be necessary as functionallity was integrated into dhs_gender_split.py on 2023-05-05

from pathlib import Path
import shutil

import pandas as pd

base = Path("/home/userx/Desktop/accessible-poverty-estimates")


src_dir = base / "data/outputs" / "GH_2014_DHS"

src_df = pd.read_csv(src_dir / "final_data.csv")
src_df.drop(columns=["Wealth Index"], inplace=True)

# src_df.loc[src_df.DHSID == "GH201400000122"]

dst_list = ["GH_2014_DHS_hoh_male", "GH_2014_DHS_hoh_female", "GH_2014_DHS_anym_male", "GH_2014_DHS_anym_female", "GH_2014_DHS_massets_male", "GH_2014_DHS_massets_female", "GH_2014_DHS_fassets_male", "GH_2014_DHS_fassets_female", "GH_2014_DHS_mf1assets_male", "GH_2014_DHS_mf1assets_female", "GH_2014_DHS_mf2assets_male", "GH_2014_DHS_mf2assets_female"]


gender_cluster_dir = base / "data" / "dhs" / "GHHR72DT"


for dst_name in dst_list:
    dst_dir = base / "data/outputs" / dst_name
    if not dst_dir.exists():
        shutil.copytree(src_dir, dst_dir)

    classification_name = dst_name.split("_")[-2]
    classification_gender = dst_name.split("_")[-1]
    gender_cluster_path = gender_cluster_dir / f"GHHR72FL_{classification_name}_{classification_gender}_cluster.csv"

    gender_cluster_df = pd.read_csv(gender_cluster_path)

    gender_cluster_df["DHSID"] =  gender_cluster_df.cluster_id.apply(lambda x: f'GH201400000{str(x).zfill(3)}')
    gender_cluster_df.rename({"wealth_index": "Wealth Index"}, axis=1, inplace=True)
    gender_cluster_df.drop(columns=["cluster_id"], inplace=True)

    new_df = src_df.merge(gender_cluster_df, on="DHSID", how="inner")

    output_path = dst_dir / "final_data.csv"
    new_df.to_csv(output_path, index=False)
