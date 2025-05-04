
from pathlib import Path
import joblib
import json

import pandas as pd
from sklearn.metrics import r2_score


base_path = Path('/home/userx/Desktop/accessible-poverty-estimates')

outputs_dir = base_path / 'data' / 'outputs'

# all households
all_model_path = outputs_dir / 'GH_2014_DHS' / 'models' / 'sub_cv5_best.joblib'
all_model = joblib.load(all_model_path)
all_data_path = outputs_dir / 'GH_2014_DHS' / 'final_data.csv'
all_data_df = pd.read_csv(all_data_path)
all_data_json_path = outputs_dir / 'GH_2014_DHS' / 'final_data.json'
all_data_json = json.load(open(all_data_json_path))
all_data_cols = all_data_json["sub_osm_cols"] + all_data_json["sub_geo_cols"]
all_data_X_test = all_data_df[all_data_cols]
all_data_Y_test = all_data_df["Wealth Index"]

# male hoh
m_model_path = outputs_dir / 'GH_2014_DHS_hoh_male' / 'models' / 'sub_cv5_best.joblib'
m_model = joblib.load(m_model_path)
m_data_path = outputs_dir / 'GH_2014_DHS_hoh_male' / 'final_data.csv'
m_data_df = pd.read_csv(m_data_path)
m_data_json_path = outputs_dir / 'GH_2014_DHS_hoh_male' / 'final_data.json'
m_data_json = json.load(open(m_data_json_path))
m_data_cols = m_data_json["sub_osm_cols"] + m_data_json["sub_geo_cols"]
m_data_X_test = m_data_df[m_data_cols]
m_data_Y_test = m_data_df["Wealth Index"]

# female hoh
f_model_path = outputs_dir / 'GH_2014_DHS_hoh_female' / 'models' / 'sub_cv5_best.joblib'
f_model = joblib.load(f_model_path)
f_data_path = outputs_dir / 'GH_2014_DHS_hoh_female' / 'final_data.csv'
f_data_df = pd.read_csv(f_data_path)
f_data_json_path = outputs_dir / 'GH_2014_DHS_hoh_female' / 'final_data.json'
f_data_json = json.load(open(f_data_json_path))
f_data_cols = f_data_json["sub_osm_cols"] + f_data_json["sub_geo_cols"]
f_data_X_test = f_data_df[f_data_cols]
f_data_Y_test = f_data_df["Wealth Index"]


# make predictions on each dataset

all_on_all_predictions = all_model.predict(all_data_X_test)
all_on_all_score = r2_score(all_on_all_predictions, all_data_Y_test)
all_on_m_predictions = all_model.predict(m_data_X_test)
all_on_m_score = r2_score(all_on_m_predictions, m_data_Y_test)
all_on_f_predictions = all_model.predict(f_data_X_test)
all_on_f_score = r2_score(all_on_f_predictions, f_data_Y_test)

m_on_all_predictions = m_model.predict(all_data_X_test)
m_on_all_score = r2_score(m_on_all_predictions, all_data_Y_test)
m_on_m_predictions = m_model.predict(m_data_X_test)
m_on_m_score = r2_score(m_on_m_predictions, m_data_Y_test)
m_on_f_predictions = m_model.predict(f_data_X_test)
m_on_f_score = r2_score(m_on_f_predictions, f_data_Y_test)

f_on_all_predictions = f_model.predict(all_data_X_test)
f_on_all_score = r2_score(f_on_all_predictions, all_data_Y_test)
f_on_m_predictions = f_model.predict(m_data_X_test)
f_on_m_score = r2_score(f_on_m_predictions, m_data_Y_test)
f_on_f_predictions = f_model.predict(f_data_X_test)
f_on_f_score = r2_score(f_on_f_predictions, f_data_Y_test)


all_data_df["all_model_predictions"] = all_on_all_predictions
all_data_df["m_model_predictions"] = m_on_all_predictions
all_data_df["f_model_predictions"] = f_on_all_predictions

m_data_df["all_model_predictions"] = all_on_m_predictions
m_data_df["m_model_predictions"] = m_on_m_predictions
m_data_df["f_model_predictions"] = f_on_m_predictions

f_data_df["all_model_predictions"] = all_on_f_predictions
f_data_df["m_model_predictions"] = m_on_f_predictions
f_data_df["f_model_predictions"] = f_on_f_predictions

all_data_df.to_csv(base_path / 'equitable-ai' /  'predictions' /'all_data.csv', index=False)
m_data_df.to_csv(base_path / 'equitable-ai' / 'predictions' / 'm_data.csv', index=False)
f_data_df.to_csv(base_path / 'equitable-ai' / 'predictions' / 'f_data.csv', index=False)

info = f"""
r2 scores for each trained model using each dataset:
- all = clusters with all households included
- m = clustering to include only male head of household
- f = clustering to include only female head of household
\n
all_on_all:\t {round(all_on_all_score, 3)}
all_on_m:\t {round(all_on_m_score, 3)}
all_on_f:\t {round(all_on_f_score, 3)}
\n
m_on_all:\t {round(m_on_all_score, 3)}
m_on_m:\t\t {round(m_on_m_score, 3)}
m_on_f:\t\t {round(m_on_f_score, 3)}
\n
f_on_all:\t {round(f_on_all_score, 3)}
f_on_m:\t\t {round(f_on_m_score, 3)}
f_on_f:\t\t {round(f_on_f_score, 3)}

"""

with open(base_path / 'equitable-ai' / 'predictions' / 'scores.txt', 'w') as f:
    f.write(info)
