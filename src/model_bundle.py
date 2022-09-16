"""

Automatically executes models.py for all projects/country groups whose output_name from config.ini is listed in an
element of the list "projects". The same hyperparameter configuration from model_utils.py is used for all projects.

"""

from configparser import ConfigParser
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
import joblib
from models import ProjectRunner

@task
def run_model(model_func):
    with joblib.parallel_backend('dask'):
        model_func()


@flow(validate_parameters=False, task_runner=DaskTaskRunner())
def run_all_projects(config: ConfigParser, project_list: list):
    for p in project_list:
        config.set(
            "main", "project", p
        )  # update config.ini to select the given country
        PR = ProjectRunner(config)
        run_model.submit(PR.run_all_osm_ntl)
        run_model.submit(PR.run_ntl)
        run_model.submit(PR.run_all_osm)
        run_model.submit(PR.run_all)
        run_model.submit(PR.run_loc)
        run_model.submit(PR.run_sub_osm_ntl)
        run_model.submit(PR.run_sub_osm)
        run_model.submit(PR.run_sub_osm_all_geo)
        run_model.submit(PR.run_all_geo)
        run_model.submit(PR.run_sub_geo)
        run_model.submit(PR.run_sub)


if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")
    project_list = [s.strip() for s in config["main"]["projects_to_run"].split(sep=",")]
    run_all_projects(config, project_list)
