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
from prefect import task, Flow, unmapped
from prefect.executors import DaskExecutor, LocalExecutor, LocalDaskExecutor


from utils import run_flow


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = config["main"]["project_dir"]
data_dir = Path(project_dir, 'data')

prefect_cloud_enabled = config.getboolean("main", "prefect_cloud_enabled")
prefect_project_name = config["main"]["prefect_project_name"]

dask_enabled = config.getboolean("main", "dask_enabled")
dask_distributed = config.getboolean("main", "dask_distributed") if "dask_distributed" in config["main"] else False

if dask_enabled:

    if dask_distributed:
        dask_address = config["main"]["dask_address"]
        executor = DaskExecutor(address=dask_address)
    else:
        executor = LocalDaskExecutor(scheduler="processes")
else:
    executor = LocalExecutor()


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


state = run_flow(flow, executor, prefect_cloud_enabled, prefect_project_name)
