from pathlib import Path
import shutil

import pandas as pd

base = Path("/home/userx/Desktop/accessible-poverty-estimates")


src_dir = base / "data/outputs" / "GH_2014_DHS"

src_df = pd.read_csv(src_dir / "dhs_data.csv")

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

    new_df = src_df.merge(gender_cluster_df, left_on="Cluster number", right_on="cluster_id", how="right")

    output_path = dst_dir / "dhs_data.csv"
    new_df.to_csv(output_path, index=False)
