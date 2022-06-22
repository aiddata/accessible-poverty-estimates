'''
Handles unzipping DHS files after they have been downloaded (manually)

Assumes that download included household recode (stata and flat formats) and gps data (flat format)

Deletes the unnecessary flat household recode files

'''
import os
import configparser
from pathlib import Path
from zipfile import ZipFile

import prefect
from prefect import task, Flow, Client
from prefect.executors import DaskExecutor, LocalExecutor
from prefect.run_configs import LocalRun


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])
data_dir = project_dir / 'data'

dask_enabled = config["main"]["dask_enabled"]
prefect_cloud_enabled = config["main"]["prefect_cloud_enabled"]


# ---------------------------------------------------------


def find_dhs_data_zip(search_dir, code, format):
    search = list(search_dir.glob(f'[A-Z][A-Z]{code}[0-9]*{format}.ZIP'))
    return search


def get_dhs_zip_iso2(file_name):
    return file_name.stem[0:2]


@task
def delete_dhs_flat_hr(data_dir):
    hr_delete = find_dhs_data_zip(data_dir, 'HR', 'FL')
    for zip in hr_delete:
        zip.unlink()



@task
def extract_dhs(data_dir):
    hr_extract_list = find_dhs_data_zip(data_dir, 'HR', 'DT')
    ge_extract_list = find_dhs_data_zip(data_dir, 'GE', 'FL')

    hr_dict = dict([(get_dhs_zip_iso2(i), i) for i in hr_extract_list])
    ge_dict = dict([(get_dhs_zip_iso2(i), i) for i in ge_extract_list])

    hr_set = set(hr_dict.keys())
    ge_set = set(ge_dict.keys())

    for i in hr_set.symmetric_difference(ge_set):
        raise Exception(f'Missing download for {i} (GE exists: {i in ge_dict}, HR exists: {i in hr_dict})')

    for zip in hr_extract_list + ge_extract_list:
        dname = zip.parent / zip.stem
        if dname.is_dir():
            print('DHS data extract exists: ', dname)
        else:
            print('Extracting DHS data: ', dname)
            with ZipFile(zip, 'r') as z:
                z.extractall(dname)


# ---------------------------------------------------------



with Flow("dhs-extract") as flow:

    delete_dhs_flat_hr(data_dir / 'dhs')

    extract_dhs(data_dir / 'dhs')


if dask_enabled:
    executor = DaskExecutor(address="tcp://127.0.0.1:8786")
else:
    executor = LocalExecutor()

# flow.run_config = LocalRun()
flow.executor = executor

if prefect_cloud_enabled:
    flow_id = flow.register(project_name="accessible-poverty-estimates")
    client = Client()
    run_id = client.create_flow_run(flow_id=flow_id)

else:
    state = flow.run()
