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
def run_project(config: ConfigParser, project: str):
    config.set(
        "main", "project", project
    )  # update config.ini to select the given country
    with joblib.parallel_backend('dask'):
        ProjectRunner(config).run_sub() # JUST RUNS SUB FOR RN


@flow(validate_parameters=False, task_runner=DaskTaskRunner())
def run_all_projects(config: ConfigParser, project_list: list):
    for p in project_list:
        run_project.submit(config, p)


if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")
    project_list = [s.strip() for s in config["main"]["projects_to_run"].split(sep=",")]
    run_all_projects(config, project_list)
