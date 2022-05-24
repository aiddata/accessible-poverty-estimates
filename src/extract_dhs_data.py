'''
Handles unzipping DHS files after they have been downloaded (manually)

Assumes that download included household recode (stata and flat formats) and gps data (flat format)

Deletes the unnecessary flat household recode files

'''
import os
import configparser
from pathlib import Path
from zipfile import ZipFile

if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')

project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

data_dir = project_dir / 'data' / 'dhs'

downloaded_files = data_dir.glob('*.ZIP')


hr_delete = list(data_dir.glob('[A-Z][A-Z]HR[0-9]*FL.ZIP'))
for zip in hr_delete:
    zip.unlink()


def get_iso2(file_name):
    return file_name.stem[0:2]


hr_extract_list = list(data_dir.glob('[A-Z][A-Z]HR[0-9]*DT.ZIP'))
ge_extract_list = list(data_dir.glob('[A-Z][A-Z]GE[0-9]*FL.ZIP'))

hr_dict = dict([(get_iso2(i), i) for i in hr_extract_list])
ge_dict = dict([(get_iso2(i), i) for i in ge_extract_list])


for i in set(hr_dict.keys()) - set(ge_dict.keys()):
    print(f'Missing download for {i} (GE exists: {i in ge_dict}, HR exists: {i in hr_dict})')

for zip in hr_extract_list + ge_extract_list:
    dname = zip.parent / zip.stem
    if not dname.is_dir():
        print(dname)
        with ZipFile(zip, 'r') as z:
            z.extractall(dname)
