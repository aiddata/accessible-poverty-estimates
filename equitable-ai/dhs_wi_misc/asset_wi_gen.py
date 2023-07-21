


import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA

from dhs_asset_components import asset_dict

household_data_path = "/home/userx/Desktop/GH_2014_DHS/GHHR72DT/GHHR72FL.DTA"
individual_data_path = "/home/userx/Desktop/GH_2014_DHS/GHIR72DT/GHIR72FL.DTA"


hr_reader = pd.read_stata(household_data_path, convert_categoricals=False, iterator=True)
hr_dict = hr_reader.variable_labels()

with hr_reader:
    hr_df = hr_reader.read()
    # hr_df.rename(columns=hr_dict, inplace=True)

hr_df['hhid'] = hr_df['hhid'].apply(lambda x: x.replace(' ', ''))


ir_reader = pd.read_stata(individual_data_path, convert_categoricals=False, iterator=False)
ir_df = ir_reader[['v001', 'v002', 'v745a']].copy()
ir_df['hhid'] = ir_df['v001'].astype(str) + ir_df['v002'].astype(str)
ir_df = ir_df[['hhid', 'v745a']].copy()
ir_df['v745a'] = ir_df['v745a'].apply(lambda x: 1 if x < 3 else 0)
ir_df = ir_df.groupby('hhid').max()['v745a'].reset_index()
hr_df = hr_df.merge(ir_df, on='hhid', how='left')




a = [i for i in asset_dict.keys() if i not in hr_df.columns.to_list()]
b = [i.split('_')[0] for i in a]
c = [i for i in b if i not in hr_df.columns.to_list()]


df = hr_df[["hhid"]].copy()


for k, v in asset_dict.items():
    print(k)
    if k in df.columns.to_list():
        continue
    try:
        print("\t...applying function")
        data = hr_df.apply(v["func"], axis=1)
        df[v["label"]] = data
    except Exception as e:
        print(v)
        raise

def memsleep(x):
    rooms = x["hv216"] if x["hv216"] > 0 else 1
    if x["hv012"] > 0:
        return x["hv012"] / rooms
    else:
        return x["hv013"] / rooms


df["Number of members per sleeping room"] = hr_df.apply(memsleep, axis=1)


asset_df = df.drop(columns=["hhid"]).copy()

for i in asset_df.columns:
    if asset_df[i].isnull().sum() > 0:
        print(i)
        asset_df[i].fillna(asset_df[i].mean(), inplace=True)

    if asset_df[i].min() == asset_df[i].max():
        print(f"dropping {i}")
        asset_df.drop(columns=[i], inplace=True)


use_pca = True

if use_pca:
    pca = PCA(1)
    pca.fit(asset_df.values)

    first_comp_vec_scaled = np.matmul(asset_df, pca.components_.T).squeeze()
    weights = pca.components_

    pd.DataFrame(list(zip(asset_df.columns, weights.squeeze())), columns=['asset', 'weight']).to_csv("/home/userx/Desktop/asset_weights.csv", index=False)

else:
    asset_df = asset_df.apply(lambda x: x - x.mean(), axis=1)
    u, s, _ = np.linalg.svd(asset_df.values.T, full_matrices=False)
    orthog_pc1_proj = np.matmul(asset_df, u[0])
    first_comp_vec_scaled = s[0] * orthog_pc1_proj


# result
print(first_comp_vec_scaled)
