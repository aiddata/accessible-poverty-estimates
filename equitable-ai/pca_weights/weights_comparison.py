import sys
from pathlib import Path

import pandas as pd


base_path = Path('/home/userx/Desktop/accessible-poverty-estimates/equitable-ai')
sys.path.insert(0, str(base_path))

from dhs_asset_components import asset_dict


all_pca_path = base_path / "pca_weights" / "all_pca.csv"
male_hoh_pca_path = base_path / "pca_weights" / "male_hoh_pca.csv"
female_hoh_pca_path = base_path / "pca_weights" / "female_hoh_pca.csv"

all_pca_df = pd.read_csv(all_pca_path)
male_hoh_pca_df = pd.read_csv(male_hoh_pca_path)
female_hoh_pca_df = pd.read_csv(female_hoh_pca_path)

all_pca_df["name"] = all_pca_df["Variable"].apply(lambda x: asset_dict[x]["label"] if x in asset_dict else x)
male_hoh_pca_df["name"] = male_hoh_pca_df["Variable"].apply(lambda x: asset_dict[x]["label"] if x in asset_dict else x)
female_hoh_pca_df["name"] = female_hoh_pca_df["Variable"].apply(lambda x: asset_dict[x]["label"] if x in asset_dict else x)

all_pca_df["pca_weight"] = all_pca_df["Comp1"].abs()
male_hoh_pca_df["pca_weight"] = male_hoh_pca_df["Comp1"].abs()
female_hoh_pca_df["pca_weight"] = female_hoh_pca_df["Comp1"].abs()

all_pca_df = all_pca_df.sort_values(by=['pca_weight'], ascending=False)
male_hoh_pca_df = male_hoh_pca_df.sort_values(by=['pca_weight'], ascending=False)
female_hoh_pca_df = female_hoh_pca_df.sort_values(by=['pca_weight'], ascending=False)

all_pca_df["rank"] = [i for i in range(1, len(all_pca_df) + 1)]
male_hoh_pca_df["rank"] = [i for i in range(1, len(male_hoh_pca_df) + 1)]
female_hoh_pca_df["rank"] = [i for i in range(1, len(female_hoh_pca_df) + 1)]

all_pca_df = all_pca_df[["name", "pca_weight", "rank"]]
male_hoh_pca_df = male_hoh_pca_df[["name", "pca_weight", "rank"]]
female_hoh_pca_df = female_hoh_pca_df[["name", "pca_weight", "rank"]]

combined_df = male_hoh_pca_df.merge(female_hoh_pca_df, on="name", suffixes=('_male', '_female'))
combined_df = all_pca_df.merge(combined_df, on="name")

combined_path = base_path / "pca_weights" / "combined_pca.csv"
combined_df.to_csv(combined_path, index=False)

import numpy as np
from sklearn.metrics import mean_squared_error

male_pca_mse = mean_squared_error(combined_df["pca_weight"], combined_df["pca_weight_male"])
male_pca_rmse = np.sqrt(male_pca_mse)
female_pca_mse = mean_squared_error(combined_df["pca_weight"], combined_df["pca_weight_female"])
female_pca_rmse = np.sqrt(female_pca_mse)

male_rank_mse = mean_squared_error(combined_df["rank"], combined_df["rank_male"])
male_rank_rmse = np.sqrt(male_rank_mse)
female_rank_mse = mean_squared_error(combined_df["rank"], combined_df["rank_female"])
female_rank_rmse = np.sqrt(female_rank_mse)

print(f"""
RMSE - All and Male PCA Weights: {round(male_pca_rmse, 5)}
RMSE - All and Male PCA Ranks: {round(male_rank_rmse, 5)}
RMSE - All and Female PCA Weights: {round(female_pca_rmse, 5)}
RMSE - All and Female PCA Ranks: {round(female_rank_rmse, 5)}
Male:Female ratio PCA Weights RMSE: {round(female_pca_rmse/male_pca_rmse,3)}
Male:Female ratio Rank RMSE: {round(female_rank_rmse/male_rank_rmse,3)}
""")


combined_df["diff_fvm"] = combined_df["pca_weight_female"] - combined_df["pca_weight_male"]
combined_df = combined_df.sort_values(by=["diff_fvm"], ascending=False)


tmp1 = combined_df.loc[(combined_df["rank"] <= 20) & (combined_df.rank_male <= 20) & (combined_df.rank_female <= 20)].copy()
tmp2 = combined_df.loc[combined_df.diff_fvm > 0.03].copy()
tmp3 = combined_df.loc[combined_df.diff_fvm < -0.03].copy()

tmp1.to_csv(base_path / "pca_weights" / "top_20_pca.csv", index=False)
tmp2.to_csv(base_path / "pca_weights" / "top_female_vs_male_pca.csv", index=False)
tmp3.to_csv(base_path / "pca_weights" / "top_male_vs_female_pca.csv", index=False)

print(f"""
Assets in top 20 PCA for all, male hoh and female HoH:\n{tmp1}
\n
Assets most influential in Female HoH PCA (vs Male HoH PCA):\n{tmp2}
\n
Assets most influential in Male HoH PCA (vs Female HoH PCA):\n{tmp3}
""")
