import sys
import os
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from mlflow import MlflowClient

from models import ProjectRunner




def parse_list(comma_sep_list: str) -> list[str]:
    return [s.strip() for s in comma_sep_list.split(sep=",")]


@task
def run_model(model_func, config):
    from models import ProjectRunner
    # initialize a ProjectRunner for this task run
    PR = ProjectRunner(config)
    # run the given model_func in our ProjectRunner
    getattr(PR, model_func)()


if __name__ == "__main__":

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

    model_funcs =  parse_list(config["main"]["model_funcs"])

    task_runner = SequentialTaskRunner


    @flow(validate_parameters=False, task_runner=task_runner)
    def run_all_projects(config: ConfigParser, project_list: list):
        # make sure the registry URI exists
        os.makedirs(config["mlflow"]["registry_uri"], exist_ok=True)

        client = MlflowClient(
            tracking_uri=config["mlflow"]["tracking_uri"],
            registry_uri=config["mlflow"]["registry_uri"],
        )

        # create an experiment with the name we chose, if it does not exist
        if not list(
            filter(
                lambda e: e.name == config["mlflow"]["experiment_name"],
                client.search_experiments(),
            )
        ):
            client.create_experiment(
                name=config["mlflow"]["experiment_name"],
                # artifact_location=config["mlflow"]["artifact_location"],
            )

        # run each model for each project
        for p in project_list:
            config.set("main", "project", p)
            for m in model_funcs:
                run_model.submit(m, config)

        # delete rogue mlruns folder in src directory
        rogue_dir = Path(config["main"]["project_dir"]) / "src" / "mlruns"
        if os.path.isdir(rogue_dir):
            rogue_dir.rmdir()


    project_list = parse_list(config["main"]["projects_to_run"])

    if config.getboolean("main", "run_sub_projects"):
        for p in project_list:
            if config.has_option(p, "sub_projects"):
                project_list.extend(parse_list(config[p]["sub_projects"]))

    run_all_projects(config, project_list)
