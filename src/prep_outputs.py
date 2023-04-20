from pathlib import Path
import shutil

base = Path("/home/userx/Desktop/accessible-poverty-estimates/data/outputs")


src = base / "GH_2014_DHS"

dst_list = ["GH_2014_DHS_hoh_male", "GH_2014_DHS_hoh_female", "GH_2014_DHS_anym_male", "GH_2014_DHS_anym_female", "GH_2014_DHS_massets_male", "GH_2014_DHS_massets_female", "GH_2014_DHS_fassets_male", "GH_2014_DHS_fassets_female", "GH_2014_DHS_mf1assets_male", "GH_2014_DHS_mf1assets_female", "GH_2014_DHS_mf2assets_male", "GH_2014_DHS_mf2assets_female"]

for dst_name in dst_list:
    dst = base / dst_name
    if not dst.exists():
        shutil.copytree(src, dst)


