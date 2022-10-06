import os
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
from dask_jobqueue import PBSCluster
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
from mlflow import MlflowClient
from models import ProjectRunner
import joblib

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

model_funcs = [
    "run_all_osm_ntl",
    "run_ntl",
    "run_all_osm",
    "run_all",
    "run_loc",
    "run_sub_osm_ntl",
    "run_sub_osm",
    "run_sub_osm_all_geo",
    "run_all_geo",
    "run_sub_geo",
    "run_sub",
]


def parse_list(comma_sep_list: str) -> list[str]:
    return [s.strip() for s in comma_sep_list.split(sep=",")]


@task
def run_model(model_func, config):
    # initialize a ProjectRunner for this task run
    PR = ProjectRunner(config)
    # run the given model_func in our ProjectRunner
    if config.getboolean("main", "use_hpc"):
        # use a threaded backend if on HPC
        with joblib.parallel_backend("threading", n_jobs=4):
            getattr(PR, model_func)()
    else:
        getattr(PR, model_func)()

if __name__ == "__main__":
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read("config.ini")

    if config.getboolean("main", "dask_enabled"):
        if config.getboolean("main", "use_hpc"):
            task_runner = DaskTaskRunner(**dask_task_runner_kwargs)
        else:
            task_runner = DaskTaskRunner
    else:
        task_runner = None

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
        (Path(config["main"]["project_dir"]) / "src" / "mlruns").rmdir()

    project_list = parse_list(config["main"]["projects_to_run"])
    for p in project_list:
        if config.has_option(p, "sub_projects"):
            project_list.extend(parse_list(config[p]["sub_projects"]))

    run_all_projects(config, project_list)
