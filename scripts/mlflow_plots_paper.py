import os
import sys
from configparser import ConfigParser, ExtendedInterpolation

from mlflow import MlflowClient
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




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

project_dir = config["main"]["project_dir"]
base_path = project_dir + "/equitable-ai/paper_figures"
os.makedirs(base_path, exist_ok=True)


client = MlflowClient(
    tracking_uri=config["mlflow"]["tracking_uri"],
    registry_uri=config["mlflow"]["registry_uri"],
)

experiments = client.search_experiments(filter_string="name = 'accessible-poverty-estimates'")
assert len(experiments) == 1

runs = client.search_runs(experiments[0].experiment_id, max_results=50000)

parent_dict_list = []
child_dict_list = []
ols_dict_list = []

for run in runs:
    print(run.info.run_id)
    run_id = run.info.run_id
    run_data = run.data
    run_tags = run_data.tags
    run_metrics = run_data.metrics

    if "mean_test_r2" not in run_metrics.keys():
        continue

    run_dict = {
        "run_id": run_id,
        "name": run_tags["mlflow.runName"]
    }

    if "version" in run_data.tags.keys():
        run_dict["version"] = run_tags["version"]


    if "mlflow.parentRunId" in run_tags:
        run_dict.update({
            "parent": run_tags["mlflow.parentRunId"],
            "max_depth": run_data.params["regressor__max_depth"],
            "max_features": run_data.params["regressor__max_features"],
            "min_samples_split": run_data.params["regressor__min_samples_split"],
            "min_samples_leaf": run_data.params["regressor__min_samples_leaf"],
            "n_estimators": run_data.params["regressor__n_estimators"],
            "r2": run_metrics["mean_test_r2"]
        })
        child_dict_list.append(run_dict)

    else:
        run_dict.update({
            "parent": "self",
            "group": run_tags["run_group"],
            "model_name": run_tags["model_name"],
            "model_type": run_tags["model_type"],
            "project": run_tags["project_name"],
        })


        name_contents = run_dict["name"].split(" - ")[0]
        if name_contents == "GH_2014_DHS":
            run_dict["gender_classification"] = "all"
            run_dict["gender"] =" all"
            run_dict["classification"] = "all"
        else:
            run_dict["gender_classification"] = name_contents.split("GH_2014_DHS_")[1]
            run_dict["gender"] = run_dict["gender_classification"].split("_")[-1]
            run_dict["classification"] = "_".join(run_dict["gender_classification"].split("_")[:-1])

        if run_tags["model_type"] == "OLS":
            metrics = run_metrics
            run_dict.update(metrics)
            ols_dict_list.append(run_dict)

        else:
            metrics = {
                "r2": run_metrics["mean_test_r2"]
            }
            run_dict["importance"] = {k:run_metrics[k] for k in run_metrics if k.endswith("importance")}

            run_dict.update(metrics)
            parent_dict_list.append(run_dict)


versions = ["1.2.0", "1.3.0", "1.4.2", "1.5.0", "1.6.1", "1.8.0"]

parent_df = pd.DataFrame(parent_dict_list)
parent_df = parent_df.loc[parent_df.version.isin(versions)].copy()

parent_df["model_and_gender"] = parent_df.model_name + "_" + parent_df.gender

# #########
parent_df = parent_df.loc[parent_df.classification.isin(["small", "medium", "all", "hoheq", "hoh"])].copy()
parent_df = parent_df.loc[parent_df.model_name.isin(["ntl", "loc", "all-geo", "sub-geo"])].copy()
# #########

child_df = pd.DataFrame(child_dict_list)
child_df = child_df.merge(parent_df[["run_id", "version", "gender", "classification"]], left_on="parent", right_on="run_id", how="left", suffixes=('', '_y'))
child_df.drop(columns=["run_id_y"], inplace=True)
child_df = child_df.loc[child_df.version.isin(versions)].copy()

child_df = child_df.loc[child_df.classification.notna()].copy()

ols_df = pd.DataFrame(ols_dict_list)

# #########
ols_df = ols_df.loc[ols_df.classification.isin(["small", "medium", "all", "hoheq", "hoh"])].copy()
ols_df = ols_df.loc[ols_df.model_name.isin(["ntl", "loc", "all-geo", "sub-geo"])].copy()
# #########

parent_path = base_path + '/parent.csv'
parent_df.to_csv(parent_path, index=False)

child_path = base_path + '/child.csv'
child_df.to_csv(child_path, index=False)

ols_path = base_path + '/ols.csv'
ols_df.to_csv(ols_path, index=False)

from collections import OrderedDict
xlabels_replace_dict = OrderedDict({
    "all": "All",
    " all": "All",
    "male": "Male",
    "female": "Female",
    "hoh": "Head of Household Gender",
    "hoh_male": "Male HoH",
    "hoh_female": "Female HoH",
    "hoheq_male": "Balanced\n Male HoH",
    "hoheq_female": "Balanced\nFemale HoH",
})

"""
"all-geo": "",
"loc": "",
"ntl": "",
"sub-geo": "",
"""


def gen_plots(dfb, cols: dict, title, tag, sort_by=None, extra_axis=True, run_grouped=True, min_yval=None):

    dfb_groups = {}
    for i in cols.keys():
        dfb_groups[i] = dfb.groupby(i, as_index=False).agg({"r2": ["min", "mean", "max", "std"]})

    for k in dfb_groups.keys():
        dfb_groups[k].columns = [k] + ["r2_min", "r2_mean", "r2_max", "r2_std"]

    plot_min_val = 0.5
    # create individual plots
    for k in dfb_groups.keys():
        tmp_df = dfb_groups[k].copy()
        if sort_by:
            tmp_df = tmp_df.sort_values(by=sort_by, ascending=True)
        data_vals = tmp_df[k].tolist()
        data = [dfb.loc[dfb[k] == i, "r2"] for i in data_vals]
        tmp_min_val = min([min(i) for i in data])
        plot_min_val = tmp_min_val if tmp_min_val < plot_min_val else plot_min_val
        fig7, ax7 = plt.subplots(1, dpi=300, figsize=(11, 7.5))
        ax7.set_title(cols[k])
        plt.title('{}: \n by {}'.format(title, cols[k]), wrap=True)
        plt.ylabel('r-squared')
        plt.xlabel(cols[k])
        if min_yval:
            ax7.set_ylim(min_yval, 0.9)
        elif plot_min_val < 0.6:
            ax7.set_ylim(0.3, 0.9)
        else:
            ax7.set_ylim(0.6, 0.9)
        ax7.boxplot(data, medianprops={'color':'black'})
        ax7.plot()
        fmt_data_vals = [xlabels_replace_dict[val] if val in xlabels_replace_dict.keys() else val for val in data_vals]
        if len(data_vals) > 3:
            ax7.set_xticklabels(fmt_data_vals, rotation=30, ha="right")
        else:
            ax7.set_xticklabels(fmt_data_vals)
        plot_path = os.path.join(base_path, "{0}_boxplot_{1}.png".format(tag, k))
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


    # create grouped plot
    if len(cols.keys()) > 1 or run_grouped:

        i_start = -1
        data_vals_grouped = []
        data_grouped = []
        xlocations = []
        second_axis_label_loc = []
        for k in dfb_groups.keys():
            data_vals = dfb_groups[k][k].tolist()
            data = [dfb.loc[dfb[k] == i, "r2"] for i in data_vals]
            i_start += 2
            xvals = []
            for i in data_vals:
                i_start += 1
                xvals.append(i_start)
            xlocations.extend(xvals)
            second_axis_label_loc.append(np.mean(xvals)-1)
            data_vals_grouped.extend(data_vals)
            data_grouped.extend(data)
        fig7, ax7 = plt.subplots(1, figsize=(18, 10))
        # plt.title(title, wrap=True)
        plt.ylabel('r-squared')
        ax7.boxplot(data_grouped, positions=xlocations, medianprops={'color':'black'})
        fmt_data_vals_grouped = [xlabels_replace_dict[val] if val in xlabels_replace_dict.keys() else val for val in data_vals_grouped]
        ax7.set_xticklabels(fmt_data_vals_grouped, rotation=30, ha="right")
        if min_yval:
            ax7.set_ylim(min_yval, 0.9)
        elif plot_min_val < 0.6:
            ax7.set_ylim(0.3, 0.9)
        else:
            ax7.set_ylim(0.6, 0.9)

        #
        if extra_axis:
            # create second Axes. Note the 0.0 height
            # ax7_position = ax7.get_position().bounds
            ax2 = fig7.add_axes((0.125, 0.03, 0.775, 0.0))
            ax2.yaxis.set_visible(False) # hide the yaxis
            ax2.set_xticks(range(max(xlocations)+1))
            ax2.set_xticks(second_axis_label_loc)
            ax2.set_xticklabels([i for i in dfb_groups.keys() if i != "sample_type"])


        k = "grouped"
        plot_path = os.path.join(base_path, "{0}_boxplot_{1}.png".format(tag, k))
        plt.savefig(plot_path)
        # pdf_plot_path = os.path.join(base_path, "{0}_boxplot_{1}.pdf".format(tag, k))
        # plt.savefig(pdf_plot_path)
        plt.close()


hp_cols = {
    "max_depth": "Max Depth",
    "max_features": "Max Features",
    "min_samples_split": "Min Split Samples",
    "min_samples_leaf": "Min Leaf Samples",
    "n_estimators": "Number of Estimators",
    "gender": "Gender",
    "classification": "Classification",
}



gen_plots(
    parent_df.loc[parent_df.version == "1.4.2"],
    {"gender": "Gender", "classification": "Classification", "model_name": "Model Name"},
    "All Classification Strategies",
    "final1", sort_by='r2_mean', extra_axis=True, run_grouped=True)

gen_plots(
    parent_df.loc[parent_df.version == "1.4.2"],
    {"gender_classification": "Gender Classification"},
    "All Classification Strategies",
    "final2", extra_axis=True, run_grouped=True)

gen_plots(
    parent_df.loc[parent_df.version == "1.4.2"],
    {"gender": "Gender", "classification": "Classification", "model_and_gender": "Model and Gender"},
    "All Classification Strategies",
    "final3", extra_axis=True, run_grouped=True)


model_gender_df = parent_df.loc[parent_df.version == "1.4.2"].copy()

# ------------------------------

model_gender_agg = model_gender_df.groupby("model_and_gender", as_index=False).agg({"r2": ["min", "mean", "max", "std"]})
model_gender_agg["model"] = model_gender_agg.model_and_gender.apply(lambda x: x.split("_")[0])
model_gender_agg["gender"] = model_gender_agg.model_and_gender.apply(lambda x: x.split("_")[1].strip())
model_gender_data_dict = {}
for ix, row in model_gender_agg.iterrows():
    model = row["model"][0]
    tmp = {
        "model": model,
        row["gender"][0]: round(row["r2"]["mean"], 3),
    }
    if model not in model_gender_data_dict:
        model_gender_data_dict[model] = {}
    model_gender_data_dict[model].update(tmp)

new_model_gender_df = pd.DataFrame(model_gender_data_dict.values())
new_model_gender_df = new_model_gender_df[["model", "all", "male", "female"]]
new_model_gender_df.sort_values('all', inplace=True, ascending=True)
new_model_gender_df.plot.bar(x="model", figsize=(10, 9))
plt.ylabel('r-squared')
plt.ylim(0.4, 0.9)
plt.xlabel('Model')
plt.xticks(rotation=30)
plot_path = os.path.join(base_path, "model_gender_bar.png")
plt.savefig(plot_path)
plt.close()

# ------------------------------



gen_plots(
    parent_df.loc[parent_df.version == "1.5.0"],
    {"gender_classification": "Gender Classification"},
    "Balanced Household Counts",
    "eq", extra_axis=True, run_grouped=True, min_yval=0.4)


all_df_set = parent_df.loc[parent_df.version == "1.6.1"].copy()
# gender_df_set = parent_df.loc[(parent_df.version == "1.5.0") & (parent_df.classification == "hoheq")].copy()
# gender_df_set.classification = gender_df_set.classification + "_" + gender_df_set.gender
# df_set = pd.concat([all_df_set, gender_df_set])
df_set = pd.concat([all_df_set])
gen_plots(
    df_set,
    {"classification": "Classification"},
    "Reduced Household Counts",
    "small", extra_axis=True, run_grouped=True, min_yval=0.5)

# ======================================================================================================================
