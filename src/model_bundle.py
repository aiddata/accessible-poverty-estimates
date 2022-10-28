import os
import json
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

import dask
from dask_jobqueue import PBSCluster
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from mlflow import MlflowClient

from models import ProjectRunner

dask.config.set({'distributed.worker.daemon': False})

cluster_kwargs = {
    "name": "ajh:ape",
    "shebang": "#!/bin/tcsh",
    "resource_spec": "nodes=1:c18a:ppn=12",
    "walltime": "00:20:00",
    "cores": 12,
    "processes": 12,
    "memory": "30GB",
    "interface": "ib0",
    "job_script_prologue": [
        "cd /sciclone/home20/jwhall/accessible-poverty-estimates/src"
    ],
    # "job_extra_directives": ["-j oe"],
}

adapt_kwargs = {
    "minimum": 12,
    "maximum": 12,
}

dask_task_runner_kwargs = {
    "cluster_class": PBSCluster,
    "cluster_kwargs": cluster_kwargs,
    "adapt_kwargs": adapt_kwargs,
}


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
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read("config.ini")

    model_funcs =  parse_list(config["main"]["model_funcs"])

    if config.getboolean("main", "dask_enabled"):
        if config.getboolean("main", "use_dask_address"):
            task_runner = DaskTaskRunner(address=config["main"]["dask_address"])
        elif config.getboolean("main", "use_hpc"):
            task_runner = DaskTaskRunner(**dask_task_runner_kwargs)
        else:
            task_runner = DaskTaskRunner
    else:
        task_runner = ConcurrentTaskRunner


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
                artifact_location=config["mlflow"]["artifact_location"],
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

    for p in project_list:
        if config.has_option(p, "sub_projects"):
            project_list.extend(parse_list(config[p]["sub_projects"]))

    run_all_projects(config, project_list)
