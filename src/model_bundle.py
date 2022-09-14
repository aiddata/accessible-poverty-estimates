"""

Automatically executes models.py for all projects/country groups whose output_name from config.ini is listed in an
element of the list "projects". The same hyperparamater configuration from model_utils.py is used for all projects.

Using os.system() because models.py is not enclosed within a function.

"""

projects = ["MULTI_SOUTH-ASIA_DHS", "BD_2017-18_DHS", "PK_2017-18_DHS", "MULTI_WEST-AFRICA_DHS", "LB_2019-20_DHS", "MR_2019-21_DHS", "SL_2019_DHS", "GN_2018_DHS", "ML_2018_DHS", "NG_2018_DHS", "BJ_2017-18_DHS", "TG_2013-14_DHS", "GH_2014_DHS", "KE_2014_DHS", "PH_2017_DHS", "ZM_2018_DHS", "TL_2016_DHS", "CM_2018_DHS"]

import configparser
import os

with open('config.ini', 'r') as savedconfigfile:
    savedconfigtext = savedconfigfile.read() #Save original config file version to restore it after execution

    config = configparser.ConfigParser()
    config.read('config.ini')

    for country in projects:
        config.set('main', 'project', country)  #Update config.ini to select the given country
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        os.system("python " + os.path.join("src", "models.py")) #Execute models.py with updated config.ini

    with open('config.ini', 'w') as newconfigfile:
        newconfigfile.write(savedconfigtext)
