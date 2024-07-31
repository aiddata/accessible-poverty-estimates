"""
python 3.9
portions of code and methodology based on https://github.com/thinkingmachines/ph-poverty-mapping
Run models based on OSM features and additional geospatial data
"""

import os
import sys
import json
from configparser import ConfigParser, ExtendedInterpolation

import mlflow
import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

import model_utils
import data_utils


class ProjectRunner:
    def __init__(self, config: ConfigParser):

        self.project = config["main"]["project"]
        self.project_dir = config["main"]["project_dir"]
        self.data_dir = os.path.join(self.project_dir, "data")

        # create dictionary of tags for this project
        self.tags = dict(config["mlflow_tags"])
        if f"{self.project}.tags" in config.sections():
            self.tags.update(dict(config[f"{self.project}.tags"]))

        # number of folds for cross-validation
        self.n_splits = 5

        self.show_plots = False

        self.output_name = config[self.project]["output_name"]

        # Scoring metrics
        self.scoring = {"r2": data_utils.pearsonr2, "rmse": data_utils.rmse, "mape": data_utils.mape}

        self.search_type = "grid"

        self.indicators = [config["main"]["indicator"]]
        # self.indicators = [
        #     'Wealth Index',
        #     'Education completed (years)',
        #     'Access to electricity',
        #     'Access to water (minutes)'
        # ]

        # make models directory
        self.models_dir = os.path.join(
            self.data_dir, "outputs", self.output_name, "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)

        # make results directory
        self.results_dir = os.path.join(
            self.data_dir, "outputs", self.output_name, "results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

        final_data_path = os.path.join(
            self.data_dir, "outputs", self.output_name, "final_data.csv"
        )

        json_path = os.path.join(
            self.data_dir, "outputs", self.output_name, "final_data.json"
        )

        self.data_df = pd.read_csv(final_data_path)

        json_data = json.load(open(json_path, "r"))

        self.all_osm_cols = json_data["all_osm_cols"]
        self.sub_osm_cols = json_data["sub_osm_cols"]
        self.all_geo_cols = json_data["all_geo_cols"]
        self.sub_geo_cols = json_data["sub_geo_cols"]
        self.ntl_cols = json_data["ntl_cols"]
        self.geom_id = json_data["primary_geom_id"]

        # set MLflow tracking location
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        sys.path.insert(0, os.path.join(self.project_dir, "src"))

    def run_OLS(self, data, y_var, x_vars, name):

        # search for this experiment id
        experiment_id = next(
            filter(
                lambda x: x.name == "accessible-poverty-estimates",
                mlflow.search_experiments(),
            ),
            None,
        ).experiment_id
        # create this run
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{self.project} - {name} - OLS",
            tags=self.tags,
        ) as run:
            # add model name to this run's tags
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("project_name", self.project)
            mlflow.set_tag("model_type", "OLS")

            # https://www.mlflow.org/docs/latest/python_api/mlflow.statsmodels.html
            mlflow.statsmodels.autolog(log_models=True)


            est = sm.OLS(endog=data[y_var], exog=sm.add_constant(data[x_vars])).fit()
            stargazer = Stargazer([est])
            with open(os.path.join(self.results_dir, f"{name}_ols.html"), "w") as f:
                f.write(stargazer.render_html())
            with open(os.path.join(self.results_dir, f"{name}_ols.tex"), "w") as f:
                f.write(stargazer.render_latex())
            return est

    def run_RF(self, name, cols):

        data_utils.plot_corr(
            data=self.data_df,
            features_cols=cols,
            indicator="Wealth Index",
            method="pearsons",
            max_n=50,
            figsize=(10, 13),
            output_file=os.path.join(self.results_dir, f"{name}_pearsons_corr.png"),
            show=self.show_plots,
        )

        data_utils.plot_corr(
            data=self.data_df,
            features_cols=cols,
            indicator="Wealth Index",
            method="spearman",
            max_n=50,
            figsize=(10, 13),
            output_file=os.path.join(self.results_dir, f"{name}_spearman_corr.png"),
            show=self.show_plots,
        )

        # search for this experiment id
        experiment_id = next(
            filter(
                lambda x: x.name == "accessible-poverty-estimates",
                mlflow.search_experiments(),
            ),
            None,
        ).experiment_id
        # create this run
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{self.project} - {name} - RF",
            tags=self.tags,
        ) as run:
            # add model name to this run's tags
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("project_name", self.project)
            mlflow.set_tag("model_type", "RF")

            mlflow.sklearn.autolog(max_tuning_runs=None)

            cv = model_utils.evaluate_model(
                data=self.data_df,
                feature_cols=cols,
                indicator_cols=self.indicators,
                clust_str=self.geom_id,
                model_name=name,
                scoring=self.scoring,
                model_type="random_forest",
                refit="r2",
                search_type=self.search_type,
                n_splits=self.n_splits,
                n_iter=10,
                plot_importance=True,
                verbose=1,
                output_file=os.path.join(
                    self.results_dir, f"{name}_model_cv{self.n_splits}_"
                ),
                show=self.show_plots,
            )

            plot_file_path = os.path.join(
                self.results_dir, f"{name}_model_grid_search_parallel_coordinates"
            )
            data_utils.plot_parallel_coordinates(
                output_file=plot_file_path,
                output_name=self.output_name,
                cv_results=cv.cv_results_,
            )

            mlflow.log_artifact(plot_file_path + ".html")


        model_utils.save_model(
            cv,
            self.data_df,
            cols,
            self.indicators,
            os.path.join(self.models_dir, f"{name}_cv{self.n_splits}_best.joblib"),
        )

    def run_model(self, name, cols, run_ols=True):
        self.run_RF(name, cols)
        if run_ols:
            self.run_OLS(self.data_df, self.indicators, cols, name)

    def run_all_osm_ntl(self):
        self.run_model("all-osm-ntl", self.all_osm_cols + self.ntl_cols)

    def run_ntl(self):
        self.run_model("ntl", self.ntl_cols)

    def run_all_osm(self):
        self.run_model("all-osm", self.all_osm_cols)

    def run_all(self):
        self.run_model("all", self.all_osm_cols + self.all_geo_cols)

    def run_loc(self):
        self.run_model("loc", ["longitude", "latitude"])

    def run_sub_osm_ntl(self):
        self.run_model("sub-osm-ntl", self.sub_osm_cols + self.ntl_cols)

    def run_sub_osm(self):
        self.run_model("sub-osm", self.sub_osm_cols)

    def run_sub_osm_all_geo(self):
        self.run_model("sub-osm-all-geo", self.sub_osm_cols + self.sub_geo_cols)

    def run_all_geo(self):
        self.run_model("all-geo", self.all_geo_cols)

    def run_sub_geo(self):
        self.run_model("sub-geo", self.sub_geo_cols)

    def run_sub(self):
        self.run_model("sub", self.sub_osm_cols + self.sub_geo_cols)

    def run_all_models(self):
        self.run_all_osm_ntl()
        self.run_ntl()
        self.run_all_osm()
        self.run_all()
        self.run_loc()
        self.run_sub_osm_ntl()
        self.run_sub_osm()
        self.run_sub_osm_all_geo()
        self.run_all_geo()
        self.run_sub_geo()
        self.run_sub()

        # final set of reduced features
        # final_features = ['longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio', 'dist_to_water_na_mean', f'viirs_{ntl_year}_median', f'viirs_{ntl_year}_max', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean']

        # final_features = [f'viirs_{ntl_year}_median', f'worldpop_pop_count_1km_mosaic_{ntl_year}_mean',  f'viirs_{ntl_year}_max', f'esa_landcover_{ntl_year}_categorical_urban', 'longitude', 'latitude', 'all_roads_length', 'all_buildings_ratio']

        """

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

        """

        # final_cv, final_predictions = run_model_funcs(final_features, 'final', **model_run_options)
        # final_ols = run_OLS(final_data_df, 'Wealth Index', final_features, 'final')

    def pop_dist_rels():
        # Explore population distribution and relationships

        data_utils.plot_hist(
            self.data_df[f"worldpop_pop_count_1km_mosaic_mean"],
            title="Distribution of Total Population",
            x_label="Total Population",
            y_label="Number of Clusters",
            output_file=os.path.join(self.results_dir, f"pop_hist.png"),
            show=self.show_plots,
        )

        data_utils.plot_regplot(
            self.data_df,
            "Wealth Index",
            "Population",
            f"worldpop_pop_count_1km_mosaic_mean",
            output_file=os.path.join(self.results_dir, f"pop_wealth_corr.png"),
            show=self.show_plots,
        )

    def ntl_mean_linear_model(self):

        # NTL mean linear model
        ntl_r2 = data_utils.plot_regplot(
            data=self.data_df,
            x_label="Wealth Index",
            y_label="Average Nightlight Intensity",
            y_var=f"viirs_mean",
            output_file=os.path.join(self.results_dir, f"ntl_wealth_regplot.png"),
            show=self.show_plots,
        )

    def adm1(self):

        adm1_df = pd.concat(
            [
                self.data_df[[geom_id] + self.indicators],
                pd.get_dummies(self.data_df["ADM1"], drop_first=True),
            ],
            axis=1,
        )
        adm1_cols = [i for i in adm1_df.columns if i not in [geom_id] + self.indicators]
        adm1_ols = run_OLS(adm1_df, "Wealth Index", adm1_cols, "adm1")

    def adm2(self):

        adm2_df = pd.concat(
            [
                self.data_df[[geom_id] + self.indicators],
                pd.get_dummies(self.data_df["ADM2"], drop_first=True),
            ],
            axis=1,
        )
        adm2_cols = [i for i in adm2_df.columns if i not in [geom_id] + self.indicators]
        adm2_ols = run_OLS(adm2_df, "Wealth Index", adm2_cols, "adm2")

    def xxx(self):

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
                self.data_df[[geom_id] + model_options["indicators"] + final_featuresx],
                pd.get_dummies(self.data_df[["ADM2"]], drop_first=True),
            ],
            axis=1,
        )
        xxx_cols = [i for i in xxx_df.columns if i not in [geom_id] + self.indicators]
        xxx_ols = run_OLS(xxx_df, "Wealth Index", xxx_cols, "adm2-final-noesa",)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.ini"

    if config_file not in os.listdir():
        raise FileNotFoundError(
            f"{config_file} file not found. Make sure you run this from the root directory of the repo and file exists."
        )

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_file)

    # ProjectRunner(config).run_all_models()
    ProjectRunner(config).run_sub()
