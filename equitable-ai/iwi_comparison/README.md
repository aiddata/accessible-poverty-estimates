# International Wealth Index (IWI) - 2014 Ghana DHS

In this folder we include two scripts used to 1) create the IWI based on the 2014 Ghana DHS, and 2) compare the wealth index produced by the DHS - the DHS Wealth Index (DHS WI) with the IWI.

Both scripts are written in Python and use the packages pandas, scipy, matplotlib. You may manage your Python environment and packages using your prefered tools such as Conda or Pip. The code is relatively simple and should function with any recent versions of Python and the packages.

Note: The code and methods used include use of the DHS "head of household gender" variable to illustrate trends that may vary along gender lines. The associated outputs may be ignored, or modified for other purposes, if you are not interested in the gender related aspects of the wealth indices.


## Data Preparation

You will need to access and download the household recode data from the 2014 Ghana DHS.
1. If you do not already have access to the data, you can [register a DHS account](https://dhsprogram.com/data/new-user-registration.cfm), login, and then create a project to [request access](https://dhsprogram.com/data/Access-Instructions.cfm) to the "Standard DHS" survey round for 2014 in Ghana.
2. Once your access to both the data is approved (may be approved at different times), use either your [account project page](https://dhsprogram.com/data/dataset_admin/index.cfm) to access the download manager.
    - In the download manager, select the Household Recode Stata dataset (.dta) for Ghana 2014 Standard DHS.
4. Once downloaded, you may wish to move the data to the same folder as this code for ease of use.
    - The Household Recode Stata dataset will typically be in a path within your download similar to `GH_2014_DHS/GHHR72DT/GHHR72FL.DTA`


## Generating IWI from 2014 Ghana DHS

To generate the IWI, first open the `gen_iwi.py` and edit the below variables in the section labeled "USER VARIABLES" defining where your input data is and where your output files will be placed. We typically recommend placing them in the same folder as this code for simplicity and defining this folder as the `base_path` variable.
    - `household_data_path`: path to DHS Household Recode Stata file
    - `output_dir`: directory where output files will be placed

You can now generate the IWI by running the script `python gen_iwi.py`. It should produce text as it calculated the IWI, and will typically take a minute or two to complete.


### Adapting for Use with Other Surveys

While the `gen_iwi.py` script was specifically created based on the 2014 Ghana DHS, you can adapt the code for use with other surveys. Adapting to other DHS surveys should be relatively straight forward. In general you will need to:
1. Download the data for the survey
2. Modify the paths in the `gen_iwi.py` script for the new data. You should make a new folder separate from this one to avoid confusing the IWI for the 2014 Ghana DHS with your other survey.
3. Update the survey variables in `gen_iwi.py` within the sections "SURVEY VARIABLES" and "SURVEY CODE".
    - If you are using another survey from the DHS Phase 7, there may be very little to modify. Other DHS Phases will likely require some modification, but it may not always be extensive. If you are using an entirely different source of survey data, this will likely require a bit of time to crosswalk the survey variables.
    - Be sure to carefully review your survey questionairre/recode documentation to make sure you get the variables correct.
    - The variables included in the "cheap utensils" and "expensive utensils" dictionaries are based on the 2014 Ghana DHS, and may need to be expanded or otherwise modified based on the what asset information is available in your survey data.



## Comparinson IWI with DHS WI

Once you have generate the IWI for your data, the resulting output file will contain IWI values for each household along with the original DHS WI values for each household. This output file can be passed to the `compare_iwi.py` script to generate a comparison between wealth estimates based on the IWI and the DHS WI.

The IWI and DHS WI do not use standardized values across both metrics, the comparisons take two forms:
1. Transition matrices based on wealth quintiles.
    - Transition matrices illustrate which wealth quintile surveyed households existing within based on the DHS WI and IWI
    - A transition matrix is generated for all households, male headed households, and female headed households. All are output in a tabular format using CSV files.
    - Additional outputs provide insight into how severe the shift between quantiles is between the DHS WI and IWI. The shift tables contain two columns in which each row defines the amount of quantiles shifted and the percentage of households which shifted that amount. These are also output in tabular CSV files
        - A summary of the shift stats is also output which provides an aggregate metric of the percentage of households which were classified in a poorer, same, or wealthier quintile using the IWI compared with the DHS WI.
2. Scatter plot comparison using normalized data.
    - Data is normalized over a range of 0 to 1 for both the DHS WI and IWI
    - The normalized DHS WI is plotted on the x-axis, and the normalized IWI is plotted on the y-axis.
    - A trend line is added running from (0,0) to (1,1)
    - By default we plot distinguish between male and female headed households using blue and red respectively.
        - An additional trend line can be added for the male and female subsets (lines may be commented out by default)

To run `compare_iwi.py` and generate the comparison:
1. Modify the `input_path` variable in the `compare_iwi.py` script to match the `output_path` variables in the `gen_iwi.py` script.
2. Set the path for the comparison outputs in `compare_iwi.py`
3. Run the script: `python compare_iwi.py`
