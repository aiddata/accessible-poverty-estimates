
import re
from pathlib import Path
import configparser

import pandas as pd
import pycountry


if 'config.ini' not in os.listdir():
    raise FileNotFoundError("config.ini file not found. Make sure you run this from the root directory of the repo.")

config = configparser.ConfigParser()
config.read('config.ini')


project = config["main"]["project"]
project_dir = Path(config["main"]["project_dir"])

min_year = 2013
max_year = 2020
osm_date = 220101
phase_list = ['DHS-VII']

# Survey Characteristics Search sorted by year
dhs_status_url = 'https://dhsprogram.com/methodology/survey-search.cfm?sendsearch=1&crt=1&listgrp=2'

status_table_list = pd.read_html(dhs_status_url)

status_df = status_table_list[-1]
status_df = status_df.rename(columns=status_df.iloc[0]).drop(status_df.index[0])

def is_junk_row(val):
    return val in map(str, range(1900,2050))

status_df.Type.apply(lambda x: is_junk_row(x))

status_df = status_df.loc[~status_df.Type.apply(lambda x: is_junk_row(x))]


status_df = status_df.loc[status_df['Type'] == 'Standard DHS']

status_df = status_df.loc[status_df['Status'] == 'Completed']
status_df = status_df.loc[status_df['Survey Datasets'] == 'Data Available']
status_df = status_df.loc[status_df['GPS Datasets'] == 'Data Available']

status_df['start_year'] = status_df['Dates of Fieldwork'].apply(lambda x: int(x.split('-')[0].strip().split('/')[-1]))
status_df['end_year'] = status_df['Dates of Fieldwork'].apply(lambda x: int(x.split('-')[1].strip().split('/')[-1]))


status_df = status_df.loc[status_df['start_year'] >= min_year]
status_df = status_df.loc[status_df['start_year'] <= max_year]

status_df['country'] = status_df['Country/Year'].apply(lambda x: re.split('(\d+)', x)[0].strip())
status_df['year_label'] = status_df.apply(lambda x: x['Country/Year'].split(x.country)[1].split('(')[0].strip(), axis=1)



def get_iso23(country):
    s = pycountry.countries.get(name=country)
    if s:
        iso2, iso3 = s.alpha_2, s.alpha_3
    else:
        try:
            fuz = pycountry.countries.search_fuzzy(country)
        except:
            return None
        if len(fuz) == 1:
            iso2, iso3 = fuz[0].alpha_2, fuz[0].alpha_3
        else:
            for f in fuz:
                if country in f.offical_name:
                    iso2, iso3 = f.alpha_2, f.alpha_3
                    break
    return iso2, iso3


phase_int_map = {
    'ix': 9,
    'viii': 8,
    'vii': 7,
    'vi': 6,
    'v': 5,
    'iv': 4,
    'iii': 3,
    'ii': 2,
    'i': 1
}


status_df[['iso2', 'iso3']] = status_df.apply(lambda x: get_iso23(x.country), axis=1, result_type='expand')

status_df.loc[status_df['country'] == 'Liberia', 'iso2'] = 'LB'
status_df.loc[status_df['country'] == 'Burundi', 'iso2'] = 'BU'
status_df.loc[status_df['country'] == 'Guatemala', 'iso2'] = 'GU'
status_df.loc[status_df['country'] == 'Timor-Leste', 'country'] = 'East Timor'

def get_file_names(row):
    if row['iso2'] is None:
        return None, None
    hr = "{}HR**DT".format( row['iso2'] )
    ge = "{}GE**FL".format( row['iso2'] )
    return hr, ge

status_df[['hr_regex', 'ge_regex']] = status_df.apply(lambda x: get_file_names(x), axis=1, result_type='expand')


status_df = status_df.loc[status_df['Phase'].isin(phase_list)]


status_df = status_df[['country', 'iso2', 'iso3', 'hr_regex', 'ge_regex', 'year_label', 'start_year', 'end_year', 'Dates of Fieldwork', 'Phase', 'Recode']]

status_df.to_csv(project_dir / 'data' / 'dhs_availability.csv', index=False)

status_df


dhs_dir = project_dir / 'data' / 'dhs'

def get_dir_exact_name(x):
    matches = list(dhs_dir.glob(x.replace('**', '*')))
    if len(matches) == 0:
        return None
    else:
        return matches[0].stem

status_df['hr_fname'] = status_df.hr_regex.apply(lambda x: get_dir_exact_name(x))
status_df['ge_fname'] = status_df.ge_regex.apply(lambda x: get_dir_exact_name(x))
status_df['prev_year'] = status_df.start_year - 1
status_df['survey_name'] = status_df.apply(lambda x: '{}_{}_DHS'.format(x.iso2, x.year_label), axis=1)

status_df = status_df.sort_values('country')


import geopandas as gpd
import pandas as pd
import utm
from pyproj import CRS

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['iso_a3', 'geometry']]

status_df = status_df.merge(world, left_on='iso3', right_on='iso_a3', how='left')

status_df = gpd.GeoDataFrame(status_df)

status_df['geometry'] = status_df.centroid


def latlon_to_utm_epsg(lat, lon):
    utm_code = utm.from_latlon(lat, lon)[2]
    south = lat < 0
    crs = CRS.from_dict({'proj': 'utm', 'zone': utm_code, 'south': south})
    epsg = crs.to_authority()[1]
    return epsg


status_df['epsg'] = status_df.geometry.apply(lambda x:latlon_to_utm_epsg(x.y, x.x))


def build_config_str(row):
    template = f'''

    [{row.survey_name}]
    output_name = {row.survey_name}
    country_name = {row.country.lower().replace(' ', '-')}
    osm_date = {osm_date}
    dhs_hh_file_name = {row.hr_fname}
    dhs_geo_file_name = {row.ge_fname}
    country_utm_epsg_code = {row.epsg}
    geom_id = DHSID
    geom_label = dhs-buffers
    geoquery_data_file_name = merge_{row.iso2.lower()}_dhs_buffer
    ntl_year = {row.prev_year}
    geospatial_variable_years = [{row.prev_year}]
    '''
    return template



status_df['config_str'] = status_df.apply(lambda x: build_config_str(x), axis=1)




_ = status_df.loc[~status_df['hr_fname'].isna(), 'config_str'].apply(lambda x: print(x))

status_df.to_csv(project_dir / 'data' / 'dhs_availability.csv', index=False)
