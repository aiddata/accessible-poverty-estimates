"""
python 3.9
portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping
Run models based on OSM features and additional geospatial data
"""

import os
import sys
import warnings
import json
from pathlib import Path
from configparser import ConfigParser

import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import mlflow

import model_utils
import data_utils


class Model:
    def __init__(
        self,
        name,
        project,
        cols,
        output_name,
        indicators,
        n_splits,
        scoring,
        search_type,
        show_plots,
        models_dir,
        results_dir,
        tags=dict(),
    ):
        self.name = name
        self.project = project
        self.cols = cols
        self.output_name = output_name
        self.indicators = indicators
        self.n_splits = n_splits
        self.scoring = scoring
        self.search_type = search_type
        self.show_plots = show_plots
        self.models_dir = models_dir
        self.results_dir = results_dir

        default_tags = {
            "project": self.project,
            "model_name": self.name,
            "output_name": self.output_name,
            "indicator": self.indicators[0],
        }
        tags.update(default_tags)
        self.tags = tags

    def run_model_funcs(self, data, geom_id):

        data_utils.plot_corr(
            data=data,
            features_cols=self.cols,
            indicator="Wealth Index",
            method="pearsons",
            max_n=50,
            figsize=(10, 13),
            output_file=os.path.join(
                self.results_dir, f"{self.name}_pearsons_corr.png"
            ),
            show=self.show_plots,
        )

        data_utils.plot_corr(
            data=data,
            features_cols=self.cols,
            indicator="Wealth Index",
            method="spearman",
            max_n=50,
            figsize=(10, 13),
            output_file=os.path.join(
                self.results_dir, f"{self.name}_spearman_corr.png"
            ),
            show=self.show_plots,
        )

        with mlflow.start_run(run_name=f"{self.project} - {self.name}") as run:

            mlflow.set_tags(self.tags)

            self.cv, self.predictions = model_utils.evaluate_model(
                data=data,
                feature_cols=self.cols,
                indicator_cols=self.indicators,
                clust_str=geom_id,
                model_name=self.name,
                scoring=self.scoring,
                model_type="random_forest",
                refit="r2",
                search_type=self.search_type,
                n_splits=self.n_splits,
                n_iter=10,
                plot_importance=True,
                verbose=1,
                output_file=os.path.join(
                    self.results_dir, f"{self.name}_model_cv{self.n_splits}_"
                ),
                show=self.show_plots,
            )

            data_utils.plot_bar_grid_search(
                output_file=os.path.join(
                    results_dir, f"{self.name}_model_grid_search_bar"
                ),
                output_name=self.output_name,
                cv_results=self.cv.cv_results_,
                grid_param="regressor__n_estimators",
            )

            plot_file_path = os.path.join(
                results_dir, f"{self.name}_model_grid_search_parallel_coordinates"
            )
            data_utils.plot_parallel_coordinates(
                output_file=plot_file_path,
                output_name=self.output_name,
                cv_results=self.cv.cv_results_,
            )

            mlflow.log_artifact(plot_file_path + ".html")

        model_utils.save_model(
            self.cv,
            data,
            self.cols,
            self.indicator,
            os.path.join(models_dir, f"{self.name}_cv{n_splits}_best.joblib"),
        )


def run_OLS(data, y_var, x_vars, name, results_dir):
    est = sm.OLS(endog=data[y_var], exog=sm.add_constant(data[x_vars])).fit()
    stargazer = Stargazer([est])
    with open(os.path.join(results_dir, f"{name}_ols.html"), "w") as f:
        f.write(stargazer.render_html())
    with open(os.path.join(results_dir, f"{name}_ols.tex"), "w") as f:
        f.write(stargazer.render_latex())
    return est


def run_models(config=ConfigParser()):

    project = config["main"]["project"]
    project_dir = config["main"]["project_dir"]
    data_dir = os.path.join(project_dir, "data")

    # create dictionary of tags for this project
    tags_dict = dict(config["mlflow_tags"])
    if f"{project}.tags" in config.sections():
        tags_dict.update(dict(config[f"{project}.tags"]))

    # these are kwargs used to initialize Model()
    model_options = {
        # number of folds for cross-validation
        "n_splits": 5,
        "show_plots": False,
        "project": project,
        "output_name": config[project]["output_name"],
        "tags": tags_dict,
        # Scoring metrics
        "scoring": {"r2": data_utils.pearsonr2, "rmse": data_utils.rmse},
        "search_type": "grid",
        "indicators": [config["main"]["indicator"]],
        # indicators = [
        #     'Wealth Index',
        #     'Education completed (years)',
        #     'Access to electricity',
        #     'Access to water (minutes)'
        # ]
    }

    # make models directory
    model_options["models_dir"] = os.path.join(
        data_dir, "outputs", model_options["output_name"], "models"
    )
    os.makedirs(model_options["models_dir"], exist_ok=True)

    # make results directory
    model_options["results_dir"] = os.path.join(
        data_dir, "outputs", model_options["output_name"], "results"
    )
    os.makedirs(model_options["results_dir"], exist_ok=True)

    final_data_path = os.path.join(
        data_dir, "outputs", model_options["output_name"], "final_data.csv"
    )

    json_path = os.path.join(
        data_dir, "outputs", model_options["output_name"], "final_data.json"
    )

    final_data_df = pd.read_csv(final_data_path)

    json_data = json.load(open(json_path, "r"))

    all_osm_cols = json_data["all_osm_cols"]
    sub_osm_cols = json_data["sub_osm_cols"]
    all_geo_cols = json_data["all_geo_cols"]
    sub_geo_cols = json_data["sub_geo_cols"]
    ntl_cols = json_data["ntl_cols"]
    geom_id = json_data["primary_geom_id"]

    # set MLflow tracking location
    mlflow.set_tracking_uri(config["main"]["mlflow_models_location"])

    sys.path.insert(0, os.path.join(project_dir, "src"))

    # -----------------------------------------------------------------------------
    # Explore population distribution and relationships

    data_utils.plot_hist(
        final_data_df[f"worldpop_pop_count_1km_mosaic_mean"],
        title="Distribution of Total Population",
        x_label="Total Population",
        y_label="Number of Clusters",
        output_file=os.path.join(model_options["results_dir"], f"pop_hist.png"),
        show=model_options["show_plots"],
    )

    data_utils.plot_regplot(
        final_data_df,
        "Wealth Index",
        "Population",
        f"worldpop_pop_count_1km_mosaic_mean",
        output_file=os.path.join(model_options["results_dir"], f"pop_wealth_corr.png"),
        show=model_options["show_plots"],
    )

    # -----------------------------------------------------------------------------

    # NTL mean linear model
    ntl_r2 = data_utils.plot_regplot(
        data=final_data_df,
        x_label="Wealth Index",
        y_label="Average Nightlight Intensity",
        y_var=f"viirs_mean",
        output_file=os.path.join(
            model_options["results_dir"], f"ntl_wealth_regplot.png"
        ),
        show=model_options["show_plots"],
    )

    adm1_df = pd.concat(
        [
            final_data_df[[geom_id] + model_options["indicators"]],
            pd.get_dummies(final_data_df["ADM1"], drop_first=True),
        ],
        axis=1,
    )
    adm1_cols = [
        i for i in adm1_df.columns if i not in [geom_id] + model_options["indicators"]
    ]
    adm1_ols = run_OLS(
        adm1_df, "Wealth Index", adm1_cols, "adm1", model_options["results_dir"]
    )

    adm2_df = pd.concat(
        [
            final_data_df[[geom_id] + model_options["indicators"]],
            pd.get_dummies(final_data_df["ADM2"], drop_first=True),
        ],
        axis=1,
    )
    adm2_cols = [
        i for i in adm2_df.columns if i not in [geom_id] + model_options["indicators"]
    ]
    adm2_ols = run_OLS(
        adm2_df, "Wealth Index", adm2_cols, "adm2", model_options["results_dir"]
    )

    # =========
    final_featuresx = [
        "viirs_median",
        "worldpop_pop_count_1km_mosaic_mean",
        "viirs_max",
        "longitude",
        "latitude",
        "all_roads_length",
        "all_buildings_ratio",
    ]

    xxx_df = pd.concat(
        [
            final_data_df[[geom_id] + model_options["indicators"] + final_featuresx],
            pd.get_dummies(final_data_df[["ADM2"]], drop_first=True),
        ],
        axis=1,
    )
    xxx_cols = [
        i for i in xxx_df.columns if i not in [geom_id] + model_options["indicators"]
    ]
    xxx_ols = run_OLS(
        xxx_df,
        "Wealth Index",
        xxx_cols,
        "adm2-final-noesa",
        model_options["results_dir"],
    )

    # =========

    model_cols = {
        "all-osm-ntl": all_osm_cols + ntl_cols,
        "ntl": ntl_cols,
        "all-osm": all_osm_cols,
        "all": all_osm_cols + all_geo_cols,
        "loc": ["longitude", "latitude"],  # location only
        "sub-osm-ntl": sub_osm_cols + ntl_cols,  # sub OSM + NTL
        "sub-osm": sub_osm_cols,
        "sub-osm-all-geo": sub_osm_cols + all_geo_cols,
        "all-geo": all_geo_cols,
        "sub-geo": sub_geo_cols,  # sub geo
        "sub": sub_osm_cols + sub_geo_cols,  # sub OSM + sub geo
    }

    models = []
    for m in model_cols.keys():
        if config.getboolean("models", m):
            M = Model(name=m, cols=model_cols[m], **model_options)
            M.run_model_funcs(final_data_df, geom_id)
            M.ols = run_OLS(
                final_data_df,
                model_options["indicator"],
                model_cols[m],
                m,
                model_options["results_dir"],
            )

    # -----------------

    # final set of reduced features
    # final_features = ['longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio', 'dist_to_water_na_mean', f'viirs_{ntl_year}_median', f'viirs_{ntl_year}_max', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean']

    # final_features = [f'viirs_{ntl_year}_median', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean',  f'viirs_{ntl_year}_max', f'esa_landcover_{ntl_year}_categorical_urban', 'longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio']

    final_features = [
        "viirs_median",
        "worldpop_pop_count_1km_mosaic_mean",
        "viirs_max",
        "esa_landcover_categorical_urban",
        "longitude",
        "latitude",
        "all_roads_length",
        "all_buildings_ratio",
    ]

    # final_cv, final_predictions = run_model_funcs(final_features, 'final', **model_run_options)
    # final_ols = run_OLS(final_data_df, 'Wealth Index', final_features, 'final', results_dir)


if __name__ == "__main__":

    if "config.ini" not in os.listdir():
        raise FileNotFoundError(
            "config.ini file not found. Make sure you run this from the root directory of the repo."
        )

    config = ConfigParser()
    config.read("config.ini")
    run_models(config)
