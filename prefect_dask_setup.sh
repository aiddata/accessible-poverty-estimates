#!/bin/bash


# ---------------------------
# if using prefect cloud, use the following to authenticate, create a project, and start your local agent
# also copy your prefect key to a ".prefect_key" file in the root directory of this repo

# in terminal 1

# set backend to cloud if using cloud management
prefect backend cloud

# auth
prefect_key=`cat .prefect_key`
prefect auth login --key $prefect_key

# create project
prefect create project accessible-poverty-estimates

# start agent
prefect agent local start

# example for starting local agent in python (not needed if using above commands)
# state = flow.run_agent(log_to_cloud=True, show_flow_logs=True)



# ---------------------------
# if using a full dask cluster

# in terminal 2
dask-scheduler

# in terminal 3
dask-worker tcp://127.0.0.1:8786 --nprocs 4 --memory-limit 4GB

# example for starting dask client in python (not needed if using above commands)
# from dask.distributed import Client
# client = Client('tcp://127.0.0.1:8786')

