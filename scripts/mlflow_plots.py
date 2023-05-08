import os
import sys
from configparser import ConfigParser, ExtendedInterpolation

from mlflow import MlflowClient
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


base_path = '/home/userx/Desktop/eqai_analysis2'
os.makedirs(base_path, exist_ok=True)


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
    run_id = run.info.run_id
    run_data = run.data
    run_tags = run_data.tags
    run_metrics = run_data.metrics

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


versions = ["1.2.0", "1.3.0", "1.4.2", "1.5.0"]

parent_df = pd.DataFrame(parent_dict_list)
parent_df = parent_df.loc[parent_df.version.isin(versions)].copy()

child_df = pd.DataFrame(child_dict_list)
child_df = child_df.merge(parent_df[["run_id", "version", "gender", "classification"]], left_on="parent", right_on="run_id", how="left", suffixes=('', '_y'))
child_df.drop(columns=["run_id_y"], inplace=True)
child_df = child_df.loc[child_df.version.isin(versions)].copy()

ols_df = pd.DataFrame(ols_dict_list)



parent_path = base_path + '/parent.csv'
parent_df.to_csv(parent_path, index=False)

child_path = base_path + '/child.csv'
child_df.to_csv(child_path, index=False)

ols_path = base_path + '/ols.csv'
ols_df.to_csv(ols_path, index=False)


xlabels_replace_dict = {
    "all": "All",
    " all": "All",
    "hoh": "Head of Household",
    "male": "Male",
    "female": "Female",
    "anym": "Has Any Males",
    "anym_male": "Has Any Males (M)",
    "anym_female": "Has Any Males (F)",
    "massets_female": "Has Male Assets (F)",
    "massets_male": "Has Male Assets (M)",
    "fassets_female": "Has Female Assets (F)",
    "fassets_male": "Has Female Assets (M)",
    "fassets": "Has Female Assets",
    "hoh_male": "Male HoH",
    "hoh_female": "Female HoH",
    "mf1assets_male": "M vs. F Assets\n(w/ overlaps)",
    "mf1assets_female": "M vs. F Assets\n(F, w/ overlaps)",
    "mf2assets_male": "M vs. F Assets\n(F, no overlaps)",
    "mf2assets_female": "M vs. F Assets\n(F, no overlaps)",

}

"""
"all-geo": "",
"all-osm": "",
"all-osm-ntl": "",
"loc": "",
"ntl": "",
"sub": "",
"sub-geo": "",
"sub-osm": "",
"sub-osm-all-geo": "",
"sub-osm-ntl": "",
"""


def gen_plots(dfb, cols: dict, visual_name, tag, extra_axis=True, run_grouped=True):

    dfb_groups = {}
    for i in cols.keys():
        dfb_groups[i] = dfb.groupby(i, as_index=False).agg({"r2": ["min", "mean", "max", "std"]})

    for k in dfb_groups.keys():
        dfb_groups[k].columns = [k] + ["r2_min", "r2_mean", "r2_max", "r2_std"]

    plot_min_val = 0.5
    # create individual plots
    for k in dfb_groups.keys():
        data_vals = dfb_groups[k][k].tolist()
        data = [dfb.loc[dfb[k] == i, "r2"] for i in data_vals]
        tmp_min_val = min([min(i) for i in data])
        plot_min_val = tmp_min_val if tmp_min_val < plot_min_val else plot_min_val
        fig7, ax7 = plt.subplots(1, dpi=300, figsize=(11, 7.5))
        ax7.set_title(cols[k])
        plt.title('Hyperparameter Results for {}: \n by {}'.format(visual_name, cols[k]), wrap=True)
        plt.ylabel('Accuracy')
        plt.xlabel(cols[k])
        if plot_min_val < 0.5:
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
        plt.title('Hyperparameter Results for {}'.format(visual_name), wrap=True)
        plt.ylabel('Accuracy')
        ax7.boxplot(data_grouped, positions=xlocations, medianprops={'color':'black'})
        fmt_data_vals_grouped = [xlabels_replace_dict[val] if val in xlabels_replace_dict.keys() else val for val in data_vals_grouped]
        ax7.set_xticklabels(fmt_data_vals_grouped, rotation=30, ha="right")
        if plot_min_val < 0.5:
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
        pdf_plot_path = os.path.join(base_path, "{0}_boxplot_{1}.pdf".format(tag, k))
        plt.savefig(pdf_plot_path)
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

gen_plots(child_df.loc[child_df.version == "1.2.0"], hp_cols, "Male vs. Female Head of Household", "hp1", extra_axis=True, run_grouped=True)
gen_plots(child_df.loc[(child_df.version == "1.2.0") & (child_df.gender == "male")], hp_cols, "Male Head of Household", "hp1m", extra_axis=True, run_grouped=True)
gen_plots(child_df.loc[(child_df.version == "1.2.0") & (child_df.gender == "female")], hp_cols, "Female Head of Household", "hp1f", extra_axis=True, run_grouped=True)

gen_plots(child_df.loc[child_df.version == "1.3.0"], hp_cols, "Presence of Males or Females in Household", "hp2", extra_axis=True, run_grouped=True)

gen_plots(parent_df.loc[parent_df.version == "1.3.0"], {"gender": "Gender", "classification": "Classification", "gender_classification": "Gender Classification", "model_name": "Model Name"}, "Presence of Males or Females in Household", "alt", extra_axis=True, run_grouped=True)

gen_plots(parent_df.loc[parent_df.version == "1.4.2"], {"gender": "Gender", "classification": "Classification", "model_name": "Model Name"}, "All Classification Strategies", "final1", extra_axis=True, run_grouped=True)
gen_plots(parent_df.loc[parent_df.version == "1.4.2"], {"gender_classification": "Gender Classification"}, "All Classification Strategies", "final2", extra_axis=True, run_grouped=True)

gen_plots(parent_df.loc[parent_df.version == "1.5.0"], {"gender_classification": "Gender Classification"}, "Balanced Household Counts", "eq", extra_axis=True, run_grouped=True)







importance_df = parent_df.loc[parent_df.version == "1.4.2"].copy()
importance_df_dict = importance_df.to_dict('records')

new_importance_df_dict = []

for row in importance_df_dict:
    row.update(row["importance"])
    del row["importance"]
    new_importance_df_dict.append(row)

new_importance_df = pd.DataFrame(new_importance_df_dict)

importance_cols = [i for i in new_importance_df.columns if i.endswith("importance")]



def plot_feat_importance(df, cols, visual_name, tag):

    all_data_items = [[i, df[i].to_numpy()] for i in cols]

    data_items = [i for i in all_data_items if max(i[1]) > 0.02]
    data_items = [[i[0], i[1][~np.isnan(i[1])]] for i in data_items]
    data_items.sort(key = lambda x: np.median(x[1]))

    data_vals = [i[1] for i in data_items]
    data_vals = [
        [ i for i in j if not pd.isnull(i)] for j in data_vals
    ]

    data_labels = [i[0].replace('_importance', '') for i in data_items]
    data_labels = [i.replace('_mean', '') if i.endswith('_mean') else i for i in data_labels]
    data_labels = [i.replace('_na', '') if i.endswith('_na') else i for i in data_labels]

    fig7, ax7 = plt.subplots(1, figsize=(30, 15))
    plt.title(visual_name)
    plt.ylabel('Importance')

    ax7.boxplot(data_vals, medianprops={'color':'black'})
    ax7.set_xticklabels(data_labels, rotation=25, ha="right")
    ax7.set_ylim(0.0, 0.7)

    plot_path = os.path.join(base_path, "feat_imp_boxplot_{}.png".format(tag))
    plt.savefig(plot_path)
    # pdf_plot_path = os.path.join(base_path, "feat_imp_boxplot_{}.pdf".format(tag))
    # plt.savefig(pdf_plot_path)


plot_feat_importance(new_importance_df, importance_cols, "Results for All Feature Importance Comparison", "all")
plot_feat_importance(new_importance_df.loc[new_importance_df.gender == "female"], importance_cols, "Results for Female Feature Importance Comparison", "female")
plot_feat_importance(new_importance_df.loc[new_importance_df.gender == "male"], importance_cols, "Results for Male Feature Importance Comparison", "male")
