# Recreating DHS Wealth Index (Stata code)

The purpose of this file is twofold. First, it investigates asset ownership rates by household gender classification. Second, it considers differences that arise between a wealth index that is created through principal component analysis of assets across all households versus wealth indices created though principal component analysis performed separately for each subsample derived from the household gender classification.


Stata do file:
- wealth_index_assets_by_classification.do


Input data:
- (required) [DHS Stata files for the Ghana 2014 survey](https://dhsprogram.com/data/dataset/Ghana_Standard-DHS_2014.cfm?flag=0)
- (optional) [DHS Shapefile for the Ghana 2014 survey cluster locations](https://dhsprogram.com/data/dataset/Ghana_Standard-DHS_2014.cfm?flag=0)
- (optional) [GADM2 Shapefile for Ghana district boundaries](https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_GHA_shp.zip)


References:
- [DHS documentation (for Wealth Index construction)](https://dhsprogram.com/programming/wealth%20index/Steps_to_constructing_the_new_DHS_Wealth_Index.pdf)



Content of Stata do file:
- Lines 1-23: File setup. User must specify the file path to the folder containing the input data described above.
- Lines 24-726: Asset variable creation, cleaning, and labeling. The asset variables are created, cleaned, and labeled following DHS documentation on the creation of the wealth index. The variables chosen for inclusion in the wealth index are based on DHS documentation. The asset variable “Boat without a motor” is listed in the documentation but not included in this do-file, as we were unable to locate this variable in the recoded data. Variable names in this do-file are based on the recoded names and may not match those names used in the DHS documentation; labels should be used to match the asset variables between this do-file and the DHS documentation.
- Lines 727-792: Calculation of asset ownership rates for subsamples derived from the household gender classifications. The classifications used for this include gender of the household head and the presence of any adult males in the household.
- Lines 793-967: Calculation of asset ownership rates, conditional on cluster fixed effects, for subsamples derived from the household gender classification. The classifications used for this include gender of the household head, the presence of any adult males in the household, and whether there are more adult males than adult females in the household.
- Lines 968-1046: (optional) Calculation of district-level variation in classifications in a format that can be used for mapping. This section is automatically turned off, as it requires access to cluster locations, which requires additional access through the DHS. If the DHS shapefile of cluster locations is obtained, the user can join cluster locations with the relevant GADM2 district polygons in QGIS by joining attributes by location and then exporting the attributes to a csv file named “GhanaDHS_GADM2.csv” in the data folder. The resulting dataset can be converted to csv and utilized in QGIS or other software to map variation in the household gender classifications across districts.
- Lines 1047-1204: Calculation of the asset wealth index based on a principal component analysis using the entire DHS sample. The steps in this section follow DHS documentation on the creation of the wealth index. The cut points used by the DHS were unknown, so quintile cut points were utilized. Some variables are dropped from the urban principal component analysis due to lack of variation; these variables are noted in the file. The resulting wealth index does not perfectly align with the assigned DHS wealth index.
- Lines 1205 -1505: Calculation of the asset wealth index based on a principal component analysis using subsamples derived from the household gender classification based on gender of the household head. The steps in this section follow DHS documentation on the creation of the wealth index, with the exception of the sample used for the principal component analysis. Some variables are dropped from the common, urban, and rural principal component analysis due to lack of variation; these variables are noted in the file.
- Lines 1506-1815: Calculation of the asset wealth index based on a principal component analysis using subsamples derived from the household gender classification based on the presence of adult males in the household. The steps in this section follow DHS documentation on the creation of the wealth index, with the exception of the sample used for the principal component analysis. Some variables are dropped from the common, urban, and rural principal component analysis due to lack of variation; these variables are noted in the file.
