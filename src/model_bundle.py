from configparser import ConfigParser
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
import mlflow
from models import ProjectRunner
import joblib

cluster_kwargs = {
    "name": "ajh:ape",
    "shebang": "#!/bin/tcsh",
    "resource_spec": "nodes=1:c18a:ppn=12",
    "walltime": "00:20:00",
    "cores": 12,
    "processes": 6,
    "memory": "12GB",
    "interface": "ib0",
    "job_script_prologue": ["cd /sciclone/home20/jwhall/accessible-poverty-estimates/src"],
    # "job_extra_directives": ["-j oe"],
}

adapt_kwargs = {
    "minimum": 36,
    "maximum": 36,
}

dask_task_runner_kwargs = {
    "cluster_class": PBSCluster,
    "cluster_kwargs": cluster_kwargs,
    "adapt_kwargs": adapt_kwargs,
}

local_cluster_kwargs = {
    "cluster_class": LocalCluster,
    "cluster_kwargs": { "n_workers": 6, },
    "adapt_kwargs": { "minimum": 4, "maximum": 12, },
}


@task
def run_model(model_func):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_func()


@flow(validate_parameters=False, task_runner=DaskTaskRunner(**dask_task_runner_kwargs))
def run_all_projects(config: ConfigParser, project_list: list):
    for p in project_list:
        config.set(
            "main", "project", p
        )  # update config.ini to select the given country
        PR = ProjectRunner(config)
        run_model.submit(PR.run_all_osm_ntl)
        run_model.submit(PR.run_ntl)
        # run_model.submit(PR.run_all_osm)
        # run_model.submit(PR.run_all)
        # run_model.submit(PR.run_loc)
        # run_model.submit(PR.run_sub_osm_ntl)
        # run_model.submit(PR.run_sub_osm)
        # run_model.submit(PR.run_sub_osm_all_geo)
        # run_model.submit(PR.run_all_geo)
        # run_model.submit(PR.run_sub_geo)
        # run_model.submit(PR.run_sub)

if __name__ == "__main__":
    config = ConfigParser()
    config.read("config.ini")
    project_list = [s.strip() for s in config["main"]["projects_to_run"].split(sep=",")]
    run_all_projects(config, project_list)
