/*******************************************************************************
File: 	wealth_index_assests_by_classification

Written by: Rachel Sayers (AidData, the College of William & Mary) 2/24/23

Updated by:	Rache Sayers (AidData, the College of William & Mary) 4/12/23

*******************************************************************************/

/*******************************************************************************
FILE SETUP
*******************************************************************************/

//setup file
clear
capture log close
log using "wealth_index_assets_by_classification.log", replace

//setup file path
//user: specify folder path containing extracted Ghana 2014 DHS Stata Data as global data
//this folder should contain the "GHIR72DT" folder within it
global data "/path/to/your/Downloads/GH_2014_DHS_xxxxxxxx_xxx_xxxxxx/"

/*******************************************************************************
DATA CLEANING
*******************************************************************************/

//The data is cleaned according to DHS documentation on the wealth index
//creation.
//The variables used are obtained from the following documentation:
//https://dhsprogram.com/programming/wealth%20index/Ghana%20DHS%202014/Ghana%20DHS%202014.xlsx
//Variable names differ from this document and are instead based on the variable
//name in the recode file.
//The method to clean the variables is based on the following documentation:
//https://dhsprogram.com/programming/wealth%20index/Steps_to_constructing_the_new_DHS_Wealth_Index.pdf
//We could not determine the exact cutpoints used in the index creation by the
//DHS and instead use the exact quintile cutpoints.

//load women's recode
cd "${data}"
cd "GHIR72DT"
use "GHIR72FL.dta", clear

//create binary variable for if any woman in a house owns a house
rename v001 hv001
rename v002 hv002
gen temp = 0
replace temp = 1 if v745a == 1 | v745a == 2 | v745a == 3
gen tag = 0
gsort hv001 hv002 -temp
by hv001 hv002: replace tag = 1 if _n == 1

keep if tag == 1
drop tag
keep hv001 hv002 temp
rename temp house_f
label var house_f "Owns a house"
cd "${data}"
save "female_house.dta", replace

//load men's recode
cd "${data}"
cd "GHMR71DT"
use "GHMR71FL.dta", clear

//create binary variable for if any man in a house owns a house
rename mv001 hv001
rename mv002 hv002
gen temp = 0
replace temp = 1 if mv745a == 1 | mv745a == 2 | mv745a == 3
gen tag = 0
gsort hv001 hv002 -temp
by hv001 hv002: replace tag = 1 if _n == 1

keep if tag == 1
drop tag
keep hv001 hv002 temp
rename temp house_m
label var house_m "Owns a house"
cd "${data}"
save "male_house.dta", replace

//load roster data
cd "${data}"
cd "GHPR72DT"
use "GHPR72FL.dta", clear

//count male permanent adult residents
gen malepermanentadult = 0
//replace malepermanentadult = 1 if hv104 == 1 & (hv105>=15 & hv105<=59) & hv102 == 1
replace malepermanentadult = 1 if hv104 == 1 & (hv105>=15 & hv105<=100) & hv102 == 1

//count female permanent adult residents
gen femalepermanentadult = 0
//replace femalepermanentadult = 1 if hv104 == 2 & (hv105>=15 & hv105<=59) & hv102 == 1
replace femalepermanentadult = 1 if hv104 == 2 & (hv105>=15 & hv105<=100) & hv102 == 1

//sum by hh
bysort hv001 hhid: egen malepermanentadult_count = total(malepermanentadult)
bysort hv001 hhid: egen femalepermanentadult_count = total(femalepermanentadult)

keep hv001 hhid hv002 malepermanentadult_count femalepermanentadult_count

rename malepermanentadult_count malepermanentadult
rename femalepermanentadult_count femalepermanentadult

gen tag = 0
bysort hv001 hhid: replace tag = 1 if _n == 1

keep if tag == 1
drop tag

cd "${data}"
cd "GHHR72DT"
merge 1:1 hv001 hhid using "GHHR72FL.dta"

drop _merge

//generate aggregate measure of house ownership
merge 1:1 hv001 hv002 using "female_house"

drop _merge

merge 1:1 hv001 hv002 using "male_house"

drop _merge

//clean components of wealth index
gen house = 0
replace house = 1 if house_m == 1 | house_f == 1
drop house_f house_m
label var house "Owns a house"

gen hv201_11 = 0
replace hv201_11 = 1 if hv201 == 11
label var hv201_11 "Source of drinking water: piped into dwelling"

gen hv201_12 = 0
replace hv201_12 = 1 if hv201 == 12
label var hv201_12 "Source of drinking water: piped into yard/plot"

gen hv201_13 = 0
replace hv201_13 = 1 if hv201 == 13
label var hv201_13 "Source of drinking water: public tap/standpipe"

gen hv201_21 = 0
replace hv201_21 = 1 if hv201 == 21
label var hv201_21 "Source of drinking water: tube well or borehole"

gen hv201_31 = 0
replace hv201_31 = 1 if hv201 == 31
label var hv201_31 "Source of drinking water: dug well protected"

gen hv201_32 = 0
replace hv201_32 = 1 if hv201 == 32
label var hv201_32 "Source of drinking water: dug well unprotected"

gen hv201_41 = 0
replace hv201_41 = 1 if hv201 == 41
label var hv201_41 "Source of drinking water: spring protected"

gen hv201_42 = 0
replace hv201_42 = 1 if hv201 == 42
label var hv201_42 "Source of drinking water: spring unprotected"

gen hv201_51 = 0
replace hv201_51 = 1 if hv201 == 51
label var hv201_51 "Source of drinking water: rainwater"

gen hv201_61 = 0
replace hv201_61 = 1 if hv201 == 61
label var hv201_61 "Source of drinking water: tanker truck"

gen hv201_81 = 0
replace hv201_81 = 1 if hv201 == 81 | hv201 == 43
label var hv201_81 "Source of drinking water: river/dam/lake/ponds/stream/canal/channel"

gen hv201_91 = 0
replace hv201_91 = 1 if hv201 == 91 | hv201 == 71
label var hv201_91 "Source of drinking water: bottled water"

gen hv201_92 = 0
replace hv201_92 = 1 if hv201 == 92 | hv201 == 72
label var hv201_92 "Source of drinking water: sachet water"

gen hv201_96 = 0
replace hv201_96 = 1 if hv201 == 96 | hv201 == 62
label var hv201_96 "Source of drinking water: other or cart with small tank"

gen hv205_11 = 0
replace hv205_11 = 1 if hv205 == 11
label var hv205_11 "Type of toilet facility: flush to piped sewer system"

gen hv205_12 = 0
replace hv205_12 = 1 if hv205 == 12
label var hv205_12 "Type of toilet facility: flush to septic tank"

gen hv205_13 = 0
replace hv205_13 = 1 if hv205 == 13
label var hv205_13 "Type of toilet facility: flush to pit latrine"

gen hv205_14 = 0
replace hv205_14 = 1 if hv205 == 14
label var hv205_14 "Type of toilet facility: flush to somewhere else"

gen hv205_15 = 0
replace hv205_15 = 1 if hv205 == 15
label var hv205_15 "Type of toilet facility: flush to don't know where"

gen hv205_21 = 0
replace hv205_21 = 1 if hv205 == 21
label var hv205_21 "Type of toilet facility: ventilated improved pit latrine"

gen hv205_22 = 0
replace hv205_22 = 1 if hv205 == 22
label var hv205_22 "Type of toilet facility: pit latrine with slab"

gen hv205_23 = 0
replace hv205_23 = 1 if hv205 == 23
label var hv205_23 "Type of toilet facility: pit latrine without slab/open pit"

gen hv205_41 = 0
replace hv205_41 = 1 if hv205 == 42
label var hv205_41 "Type of toilet facility: bucket toilet"

gen hv205_51 = 0
replace hv205_51 = 1 if hv205 == 51 | hv205 == 43
label var hv205_51 "Type of toilet facility: hanging toilet/hanging latrine"

gen hv205_61 = 0
replace hv205_61 = 1 if hv205 == 61 | hv205 == 31
label var hv205_61 "Type of toilet facility: no facility/bush/field"

gen hv205_11_sh = 0
replace hv205_11_sh = 1 if hv205 == 11 & (hv225 == 1 | hv225 == 2)
label var hv205_11_sh "Type of toilet facility: flush to piped sewer system shared"

gen hv205_12_sh = 0
replace hv205_12_sh = 1 if hv205 == 12 & (hv225 == 1 | hv225 == 2)
label var hv205_12_sh "Type of toilet facility: flush to septic tank shared"

gen hv205_13_sh = 0
replace hv205_13_sh = 1 if hv205 == 13 & (hv225 == 1 | hv225 == 2)
label var hv205_13_sh "Type of toilet facility: flush to pit latrine shared"

gen hv205_14_sh = 0
replace hv205_14_sh = 1 if hv205 == 14 & (hv225 == 1 | hv225 == 2)
label var hv205_14_sh "Type of toilet facility: flush to somewhere else shared"

gen hv205_15_sh = 0
replace hv205_15_sh = 1 if hv205 == 15 & (hv225 == 1 | hv225 == 2)
label var hv205_15_sh "Type of toilet facility: flush to don't know where shared"

gen hv205_21_sh = 0
replace hv205_21_sh = 1 if hv205 == 21 & (hv225 == 1 | hv225 == 2)
label var hv205_21_sh "Type of toilet facility: ventilated improved pit latrine shared"

gen hv205_22_sh = 0
replace hv205_22_sh = 1 if hv205 == 22 & (hv225 == 1 | hv225 == 2)
label var hv205_22_sh "Type of toilet facility: pit latrine with slab shared"

gen hv205_23_sh = 0
replace hv205_23_sh = 1 if hv205 == 23 & (hv225 == 1 | hv225 == 2)
label var hv205_23_sh "Type of toilet facility: pit latrine without slab/open pit shared"

gen hv205_41_sh = 0
replace hv205_41_sh = 1 if hv205 == 42 & (hv225 == 1 | hv225 == 2)
label var hv205_41_sh "Type of toilet facility: bucket toilet shared"

gen hv205_51_sh = 0
replace hv205_51_sh = 1 if (hv205 == 51 | hv205 == 43) & (hv225 == 1 | hv225 == 2)
label var hv205_51_sh "Type of toilet facility: hanging toilet/hanging latrine shared"

label var hv206 "Electricity"
replace hv206 = 0 if hv206 == .
label var sh110b "Wall clock"
replace sh110b = 0 if sh110b == .
label var hv207 "Radio"
replace hv207 = 0 if hv207 == .
label var sh110d "Black/white television"
replace sh110d = 0 if sh110d == .
label var hv208 "Color television"
replace hv208 = 0 if hv208 == .
label var hv243a "Mobile telephone"
replace hv243a = 0 if hv243a == .
label var hv221 "Telephone (non-mobile)"
replace hv221 = 0 if hv221 == .
label var hv209 "Refrigerator"
replace hv209 = 0 if hv209 == .
label var sh110i "Freezer"
replace sh110i = 0 if sh110i == .
label var sh110j "Electric generator/inverter"
replace sh110j = 0 if sh110j == .
label var sh110k "Washing machine"
replace sh110k = 0 if sh110k == .
label var sh110l "Computer/tablet computer"
replace sh110l = 0 if sh110l == .
label var sh110m "Photo camera"
replace sh110m = 0 if sh110m == .
label var sh110n "Video deck/DVD/VCD"
replace sh110n = 0 if sh110n == .
label var sh110o "Sewing machine"
replace sh110o = 0 if sh110o == .
label var sh110p "Bed"
replace sh110p = 0 if sh110p == .
label var sh110q "Table"
replace sh110q = 0 if sh110q == .
label var sh110r "Cabinet/cupboard"
replace sh110r = 0 if sh110r == .
label var sh110s "Access to internet in any device"
replace sh110s = 0 if sh110s == .

gen hv226_1 = 0
replace hv226_1 = 1 if hv226 == 1
label var hv226_1 "Type of cooking fuel: electricity"

gen hv226_2 = 0
replace hv226_2 = 1 if hv226 == 2
label var hv226_2 "Type of cooking fuel: LPG"

gen hv226_3 = 0
replace hv226_3 = 1 if hv226 == 3
label var hv226_3 "Type of cooking fuel: natural gas"

gen hv226_5 = 0
replace hv226_5 = 1 if hv226 == 5
label var hv226_5 "Type of cooking fuel: kerosene"

gen hv226_6 = 0
replace hv226_6 = 1 if hv226 == 6
label var hv226_6 "Type of cooking fuel: coal/lignite"

gen hv226_7 = 0
replace hv226_7 = 1 if hv226 == 7
label var hv226_7 "Type of cooking fuel: charcoal"

gen hv226_8 = 0
replace hv226_8 = 1 if hv226 == 8
label var hv226_8 "Type of cooking fuel: wood"

gen hv226_9 = 0
replace hv226_9 = 1 if hv226 == 9
label var hv226_9 "Type of cooking fuel: straw/shrubs/grass"

gen hv226_10 = 0
replace hv226_10 = 1 if hv226 == 10
label var hv226_10 "Type of cooking fuel: agricultural crop"

gen hv226_95 = 0
replace hv226_95 = 1 if hv226 == 95
label var hv226_95 "Type of cooking fuel: no food cooked in house"

gen hv213_11 = 0
replace hv213_11 = 1 if hv213 == 11
label var hv213_11 "Main floor material: earth/sand"

gen hv213_12 = 0
replace hv213_12 = 1 if hv213 == 12
label var hv213_12 "Main floor material: dung"

gen hv213_21 = 0
replace hv213_21 = 1 if hv213 == 21
label var hv213_21 "Main floor material: wood planks"

gen hv213_31 = 0
replace hv213_31 = 1 if hv213 == 31
label var hv213_31 "Main floor material: parquet/polished wood"

gen hv213_33 = 0
replace hv213_33 = 1 if hv213 == 33
label var hv213_33 "Main floor material: ceramic/marble/porcelain tiles/terrazo"

gen hv213_34 = 0
replace hv213_34 = 1 if hv213 == 34
label var hv213_34 "Main floor material: cement"

gen hv213_35 = 0
replace hv213_35 = 1 if hv213 == 35
label var hv213_35 "Main floor material: woolen carpets/synthetic carpet"

gen hv213_36 = 0
replace hv213_36 = 1 if hv213 == 36
label var hv213_36 "Main floor material: linoleum/rubber carpet"

gen hv213_96 = 0
replace hv213_96 = 1 if hv213 == 96
label var hv213_96 "Main floor material: other"

gen hv215_11 = 0
replace hv215_11 = 1 if hv215 == 11
label var hv215_11 "Main roof material: no roof"

gen hv215_12 = 0
replace hv215_12 = 1 if hv215 == 12
label var hv215_12 "Main roof material: thatch/palm leaf"

gen hv215_21 = 0
replace hv215_21 = 1 if hv215 == 21
label var hv215_21 "Main roof material: rustic mat"

gen hv215_22 = 0
replace hv215_22 = 1 if hv215 == 22
label var hv215_22 "Main roof material: palm/bamboo"

gen hv215_23 = 0
replace hv215_23 = 1 if hv215 == 23
label var hv215_23 "Main roof material: wood planks"

gen hv215_31 = 0
replace hv215_31 = 1 if hv215 == 31
label var hv215_31 "Main roof material: metal"

gen hv215_32 = 0
replace hv215_32 = 1 if hv215 == 32
label var hv215_32 "Main roof material: wood"

gen hv215_33 = 0
replace hv215_33 = 1 if hv215 == 33
label var hv215_33 "Main roof material: calamine/cement fiber"

gen hv215_34 = 0
replace hv215_34 = 1 if hv215 == 34
label var hv215_34 "Main roof material: ceramic tiles"

gen hv215_35 = 0
replace hv215_35 = 1 if hv215 == 35
label var hv215_35 "Main roof material: cement"

gen hv215_36 = 0
replace hv215_36 = 1 if hv215 == 36
label var hv215_36 "Main roof material: roofing shingles"

gen hv215_37 = 0
replace hv215_37 = 1 if hv215 == 37
label var hv215_37 "Main roof material: asbestos/slate roofing sheets"

gen hv215_96 = 0
replace hv215_96 = 1 if hv215 == 96
label var hv215_96 "Main roof material: other"

gen hv214_11 = 0
replace hv214_11 = 1 if hv214 == 11
label var hv214_11 "Main wall material: no walls"

gen hv214_12 = 0
replace hv214_12 = 1 if hv214 == 12
label var hv214_12 "Main wall material: cane/palm/trunks"

gen hv214_13 = 0
replace hv214_13 = 1 if hv214 == 13
label var hv214_13 "Main wall material: dirt"

gen hv214_21 = 0
replace hv214_21 = 1 if hv214 == 21
label var hv214_21 "Main wall material: bamboo with mud"

gen hv214_22 = 0
replace hv214_22 = 1 if hv214 == 22
label var hv214_22 "Main wall material: stone with mud"

gen hv214_23 = 0
replace hv214_23 = 1 if hv214 == 23
label var hv214_23 "Main wall material: uncovered adobe"

gen hv214_24= 0
replace hv214_24 = 1 if hv214 == 24
label var hv214_24 "Main wall material: plywood"

gen hv214_26 = 0
replace hv214_26 = 1 if hv214 == 26
label var hv214_26 "Main wall material: reused wood"

gen hv214_31 = 0
replace hv214_31 = 1 if hv214 == 31
label var hv214_31 "Main wall material: cement"

gen hv214_32 = 0
replace hv214_32 = 1 if hv214 == 32
label var hv214_32 "Main wall material: stone with lime/cement"

gen hv214_33 = 0
replace hv214_33 = 1 if hv214 == 33
label var hv214_33 "Main wall material: bricks"

gen hv214_34 = 0
replace hv214_34 = 1 if hv214 == 34
label var hv214_34 "Main wall material: cement blocks"

gen hv214_35 = 0
replace hv214_35 = 1 if hv214 == 35
label var hv214_35 "Main wall material: covered adobe"

gen hv214_36 = 0
replace hv214_36 = 1 if hv214 == 36
label var hv214_36 "Main wall material: wood planks/shingles"

gen hv214_96 = 0
replace hv214_96 = 1 if hv214 == 96
label var hv214_96 "Main wall material: other"

label var hv243b "Watch"
replace hv243b = 0 if hv243b == .
label var hv210 "Bicycle"
replace hv210 = 0 if hv210 == .
label var hv211 "Motorcycle/scooter"
replace hv211 = 0 if hv211 == .
label var hv243c "Animal-drawn cart"
replace hv243c = 0 if hv243c == .
label var hv212 "Car/truck"
replace hv212 = 0 if hv212 == .
label var hv243d "Boat with a motor"
replace hv243d = 0 if hv243d == .
label var hv247 "Bank account"
replace hv247 = 0 if hv247 == .
gen land = hv244
label var land "Owns land suitable for agriculture"
replace land = 0 if hv244 == .

gen memsleep = .
replace memsleep = hv012/hv216 if hv216 !=0
replace memsleep = hv012 if hv216==0

label var memsleep "Number of members per sleeping room"

//define common component list
local assetlist = "hv201_11 hv201_12 hv201_13 hv201_21 hv201_31 hv201_32 hv201_41 hv201_42 hv201_51 hv201_61 hv201_81 hv201_91 hv201_92 hv201_96 hv205_11 hv205_12 hv205_13 hv205_14 hv205_15 hv205_21 hv205_22 hv205_23 hv205_41 hv205_51 hv205_61 hv205_11_sh hv205_12_sh hv205_13_sh hv205_14_sh hv205_15_sh hv205_21_sh hv205_22_sh hv205_23_sh hv205_41_sh hv205_51_sh hv206 sh110b hv207 sh110d hv208 hv243a hv221 hv209 sh110i sh110j sh110k sh110l sh110m sh110n sh110o sh110p sh110q sh110r sh110s hv226_1 hv226_2 hv226_3 hv226_5 hv226_6 hv226_7 hv226_8 hv226_9 hv226_10 hv226_95 hv213_11 hv213_12 hv213_21 hv213_31 hv213_33 hv213_34 hv213_35 hv213_36 hv213_96 hv215_11 hv215_12 hv215_21 hv215_22 hv215_23 hv215_31 hv215_32 hv215_33 hv215_34 hv215_35 hv215_36 hv215_37 hv215_96 hv214_11 hv214_12 hv214_13 hv214_21 hv214_22 hv214_23 hv214_24 hv214_26 hv214_31 hv214_32 hv214_33 hv214_34 hv214_35 hv214_36 hv214_96 hv243b hv210 hv211 hv243c hv212 hv243d hv247 house land memsleep"

//clean rural assets

gen landarea = hv245
replace landarea = 0 if hv244 == 0
replace landarea = . if landarea == 998 & hv244 == 1

gen hv246a_0 = 0
label var hv246a_0 "Cattle: None"
replace hv246a_0 = 1 if hv246a == 0

gen hv246a_1 = 0
label var hv246a_1 "Cattle: 1-4"
replace hv246a_1 = 1 if hv246a >= 1 & hv246a <= 4

gen hv246a_2 = 0
label var hv246a_2 "Cattle: 5-9"
replace hv246a_2 = 1 if hv246a >= 5 & hv246a <= 9

gen hv246a_3 = 0
label var hv246a_3 "Cattle: 10+"
replace hv246a_3 = 1 if hv246a >= 10 & hv246a <= 95

gen hv246b_0 = 0
label var hv246b_0 "Cows/bulls: None"
replace hv246b_0 = 1 if hv246b == 0

gen hv246b_1 = 0
label var hv246b_1 "Cows/bulls: 1-4"
replace hv246b_1 = 1 if hv246b >= 1 & hv246b <= 4

gen hv246b_2 = 0
label var hv246b_2 "Cows/bulls: 5-9"
replace hv246b_2 = 1 if hv246b >= 5 & hv246b <= 9

gen hv246b_3 = 0
label var hv246b_3 "Cows/bulls: 10+"
replace hv246b_3 = 1 if hv246b >= 10 & hv246b <= 95

gen hv246c_0 = 0
label var hv246c_0 "Horses/donkeys/mules: None"
replace hv246c_0 = 1 if hv246c == 0

gen hv246c_1 = 0
label var hv246c_1 "Horses/donkeys/mules: 1-4"
replace hv246c_1 = 1 if hv246c >= 1 & hv246c <= 4

gen hv246c_2 = 0
label var hv246c_2 "Horses/donkeys/mules: 5+"
replace hv246c_2 = 1 if hv246c >= 5 & hv246c <= 95

gen hv246d_0 = 0
label var hv246d_0 "Goats: None"
replace hv246d_0 = 1 if hv246d == 0

gen hv246d_1 = 0
label var hv246d_1 "Goats: 1-4"
replace hv246d_1 = 1 if hv246d >= 1 & hv246d <= 4

gen hv246d_2 = 0
label var hv246d_2 "Goats: 5-9"
replace hv246d_2 = 1 if hv246d >= 5 & hv246d <= 9

gen hv246d_3 = 0
label var hv246d_3 "Goats: 10+"
replace hv246d_3 = 1 if hv246d >= 10 & hv246d <= 95

gen hv246g_0 = 0
label var hv246g_0 "Pigs: None"
replace hv246g_0 = 1 if hv246g == 0

gen hv246g_1 = 0
label var hv246g_1 "Pigs: 1-4"
replace hv246g_1 = 1 if hv246g >= 1 & hv246g <= 4

gen hv246g_2 = 0
label var hv246g_2 "Pigs: 5-9"
replace hv246g_2 = 1 if hv246g >= 5 & hv246g <= 9

gen hv246g_3 = 0
label var hv246g_3 "Pigs: 10+"
replace hv246g_3 = 1 if hv246g >= 10 & hv246g <= 95

gen hv246h_0 = 0
label var hv246h_0 "Rabbits: None"
replace hv246h_0 = 1 if hv246h == 0

gen hv246h_1 = 0
label var hv246h_1 "Rabbits: 1-4"
replace hv246h_1 = 1 if hv246h >= 1 & hv246h <= 4

gen hv246h_2 = 0
label var hv246h_2 "Rabbits: 5-9"
replace hv246h_2 = 1 if hv246h >= 5 & hv246h <= 9

gen hv246h_3 = 0
label var hv246h_3 "Rabbits: 10+"
replace hv246h_3 = 1 if hv246h >= 10 & hv246h <= 95

gen hv246i_0 = 0
label var hv246i_0 "Grasscutters: None"
replace hv246i_0 = 1 if hv246i == 0

gen hv246i_1 = 0
label var hv246i_1 "Grasscutters: 1-4"
replace hv246i_1 = 1 if hv246i >= 1 & hv246i <= 4

gen hv246i_2 = 0
label var hv246i_2 "Grasscutters: 5-9"
replace hv246i_2 = 1 if hv246i >= 5 & hv246i <= 9

gen hv246i_3 = 0
label var hv246i_3 "Grasscutters: 10+"
replace hv246i_3 = 1 if hv246i >= 10 & hv246i <= 95

gen hv246e_0 = 0
label var hv246e_0 "Sheep: None"
replace hv246e_0 = 1 if hv246e == 0

gen hv246e_1 = 0
label var hv246e_1 "Sheep: 1-4"
replace hv246e_1 = 1 if hv246e >= 1 & hv246e <= 4

gen hv246e_2 = 0
label var hv246e_2 "Sheep: 5-9"
replace hv246e_2 = 1 if hv246e >= 5 & hv246e <= 9

gen hv246e_3 = 0
label var hv246e_3 "Sheep: 10+"
replace hv246e_3 = 1 if hv246e >= 10 & hv246e <= 95

gen hv246f_0 = 0
label var hv246f_0 "Chickens: None"
replace hv246f_0 = 1 if hv246f == 0

gen hv246f_1 = 0
label var hv246f_1 "Chickens: 1-9"
replace hv246f_1 = 1 if hv246f >= 1 & hv246f <= 9

gen hv246f_2 = 0
label var hv246f_2 "Chickens: 10-29"
replace hv246f_2 = 1 if hv246f >= 10 & hv246f <= 29

gen hv246f_3 = 0
label var hv246f_3 "Chickens: 30+"
replace hv246f_3 = 1 if hv246f >= 30 & hv246f <= 95

gen hv246j_0 = 0
label var hv246j_0 "Other poultry: None"
replace hv246j_0 = 1 if hv246j == 0

gen hv246j_1 = 0
label var hv246j_1 "Other poultry: 1-9"
replace hv246j_1 = 1 if hv246j >= 1 & hv246j <= 9

gen hv246j_2 = 0
label var hv246j_2 "Other poultry: 10-29"
replace hv246j_2 = 1 if hv246j >= 10 & hv246j <= 29

gen hv246j_3 = 0
label var hv246j_3 "Other poultry: 30+"
replace hv246j_3 = 1 if hv246j >= 30 & hv246j <= 95

gen hv246k_0 = 0
label var hv246k_0 "Sheep: None"
replace hv246k_0 = 1 if hv246k == 0

gen hv246k_1 = 0
label var hv246k_1 "Sheep: 1-4"
replace hv246k_1 = 1 if hv246k >= 1 & hv246k <= 4

gen hv246k_2 = 0
label var hv246k_2 "Sheep: 5-9"
replace hv246k_2 = 1 if hv246k >= 5 & hv246k <= 9

gen hv246k_3 = 0
label var hv246k_3 "Sheep: 10+"
replace hv246k_3 = 1 if hv246k >= 10 & hv246k <= 95

//define rural asset list
local ruralassetlist = "hv201_11 hv201_12 hv201_13 hv201_21 hv201_31 hv201_32 hv201_41 hv201_42 hv201_51 hv201_61 hv201_81 hv201_91 hv201_92 hv201_96 hv205_11 hv205_12 hv205_13 hv205_14 hv205_21 hv205_22 hv205_23 hv205_41 hv205_51 hv205_61 hv205_11_sh hv205_12_sh hv205_13_sh hv205_15_sh hv205_21_sh hv205_22_sh hv205_23_sh hv205_41_sh hv205_51_sh hv206 sh110b hv207 sh110d hv208 hv243a hv221 hv209 sh110i sh110j sh110k sh110l sh110m sh110n sh110o sh110p sh110q sh110r sh110s hv226_1 hv226_2 hv226_3 hv226_5 hv226_6 hv226_7 hv226_8 hv226_9 hv226_10 hv226_95 hv213_11 hv213_12 hv213_31 hv213_33 hv213_34 hv213_35 hv213_36 hv213_96 hv215_11 hv215_12 hv215_21 hv215_22 hv215_23 hv215_31 hv215_32 hv215_33 hv215_34 hv215_35 hv215_36 hv215_37 hv215_96 hv214_11 hv214_12 hv214_13 hv214_21 hv214_22 hv214_23 hv214_24 hv214_26 hv214_31 hv214_32 hv214_33 hv214_34 hv214_35 hv214_36 hv214_96 hv243b hv210 hv211 hv243c hv212 hv243d hv247 house land memsleep landarea hv246a_0 hv246a_1 hv246a_2 hv246a_3 hv246b_0 hv246b_1 hv246b_2 hv246b_3 hv246c_0 hv246c_1 hv246c_2 hv246d_0 hv246d_1 hv246d_2 hv246d_3 hv246g_0 hv246g_1 hv246g_2 hv246g_3 hv246h_0 hv246h_1 hv246h_2 hv246h_3 hv246i_0 hv246i_1 hv246i_2 hv246i_3 hv246e_0 hv246e_1 hv246e_2 hv246e_3 hv246f_0 hv246f_1 hv246f_2 hv246f_3 hv246j_0 hv246j_1 hv246j_2 hv246j_3 hv246k_0 hv246k_1 hv246k_2 hv246k_3"

//define urban asset list
local urbanassetlist = "hv201_11 hv201_12 hv201_13 hv201_21 hv201_31 hv201_32 hv201_41 hv201_42 hv201_51 hv201_61 hv201_81 hv201_91 hv201_92 hv201_96 hv205_11 hv205_12 hv205_13 hv205_14 hv205_15 hv205_21 hv205_22 hv205_23 hv205_41 hv205_51 hv205_61 hv205_11_sh hv205_12_sh hv205_13_sh hv205_14_sh hv205_15_sh hv205_21_sh hv205_22_sh hv205_23_sh hv205_41_sh hv205_51_sh hv206 sh110b hv207 sh110d hv208 hv243a hv221 hv209 sh110i sh110j sh110k sh110l sh110m sh110n sh110o sh110p sh110q sh110r sh110s hv226_1 hv226_2 hv226_3 hv226_5 hv226_6 hv226_7 hv226_8 hv226_9 hv226_10 hv226_95 hv213_11 hv213_12 hv213_21 hv213_31 hv213_33 hv213_34 hv213_35 hv213_36 hv213_96 hv215_11 hv215_12 hv215_21 hv215_22 hv215_23 hv215_31 hv215_32 hv215_33 hv215_34 hv215_35 hv215_36 hv215_37 hv215_96 hv214_11 hv214_12 hv214_13 hv214_21 hv214_22 hv214_23 hv214_24 hv214_26 hv214_31 hv214_32 hv214_33 hv214_34 hv214_35 hv214_36 hv214_96 hv243b hv210 hv211 hv243c hv212 hv243d hv247 house land memsleep landarea hv246a_0 hv246a_1 hv246a_2 hv246a_3 hv246b_0 hv246b_1 hv246b_2 hv246b_3 hv246c_0 hv246c_1 hv246c_2 hv246d_0 hv246d_1 hv246d_2 hv246d_3 hv246g_0 hv246g_1 hv246g_2 hv246g_3 hv246h_0 hv246h_1 hv246h_2 hv246h_3 hv246i_0 hv246i_1 hv246i_2 hv246i_3 hv246e_0 hv246e_1 hv246e_2 hv246e_3 hv246f_0 hv246f_1 hv246f_2 hv246f_3 hv246j_0 hv246j_1 hv246j_2 hv246j_3 hv246k_0 hv246k_1 hv246k_2 hv246k_3"

//tag hhs will male hoh
gen malehoh = 0
replace malehoh = 1 if hv219 == 1
label var malehoh "Male Head of Household"

//tag hhs with 1+ male permanent adult (15-100) in hh
gen maleadultpres = 0
replace maleadultpres = 1 if malepermanentadult > 0
label var maleadultpres "Male Adult in Household"

//tag hhs with more male than female permanent adults (15-100) in hh
gen moremales = 0
replace moremales = 1 if malepermanentadult > femalepermanentadult
label var moremales "More Male Adults than Female in Household"

/*******************************************************************************
RATES OF ASSET OWNERSHIP BY HH CLASSIFICATION, UNCONDITIONAL MEANS
*******************************************************************************/

//determine rates of ownership for each asset by each HH classification
//the two classificaitons used are gender of household head and whether any
//adult males are in the household

cd "${data}"

//loop by classification
forvalues i = 1/2 {
	//set up row labels
	if `i' == 1 {
		putexcel set "asset_means_by_hohgender", replace
		putexcel C1 = "% Among HHs w/ Female HoH"
		putexcel D1 = "% Among HHs w/ Male HoH"
	}
	else {
		putexcel set "asset_means_by_anymales", replace
		putexcel C1 = "% Among HHs w/ No Males"
		putexcel D1 = "% Among HHs w/ 1+ Males"
	}
	putexcel B1 = "% Among All HHs"
	putexcel E1 = "Difference (Female-Male)"

	local count = 1
	//loop through assets
	foreach asset of local assetlist {
		local count = `count' + 1
		if `i' == 1 {
			local samp = "malehoh"
		}
		else {
			local samp = "maleadultpres"
		}
		//calculate asset rates by hh classification
		summ `asset'
		local all: display %6.2f r(mean)*100
		summ `asset' if `samp' == 0
		local female: display %6.2f r(mean)*100
		summ `asset' if `samp' == 1
		local male: display %6.2f r(mean)*100
		local diff: display %6.2f `female' - `male'
		ttest `asset', by(`samp')
		local pvalue = r(p)
		local star = ""
		if `pvalue' <.1 {
			local star = "`star'*"
			if `pvalue' <.05 {
				local star = "`star'*"
				if `pvalue' <.01 {
					local star = "`star'*"
				}
			}
		}
		//store results in spreadsheet
		local label: variable label `asset'
		putexcel A`count' = "`label'"
		putexcel C`count' = "`female'"
		putexcel D`count' = "`male'"
		putexcel B`count' = "`all'"
		putexcel E`count' = "`diff'`star'"
	}
}

/*******************************************************************************
RATES OF ASSET OWNERSHIP BY HH CLASSIFICATION, CONDITIONAL MEANS, CLUSTER FEs
*******************************************************************************/

//determine rates of ownership for each asset by each HH classification,
//conditional on cluster FEs
//the three classifications used are gender of household head, whether any
//adult males are in the household, and whether there are more adult males
//than adult females in the household

//determine rates of ownership for each asset by each HH classification -
//conditional means, cluster FEs, SEs clustered by cluster

local categories = "malehoh maleadultpres moremales"
gen asset = 0
//loop by classification
foreach c of local categories {
	local n = 0
	//loop through assets
	foreach a of local assetlist {
		local label : variable label `a'
		local n = `n' + 1
		gen `c'`n' = `c'
		label var `c'`n' "`label'"
		replace asset = `a'
		//calculate conditional rates of asset ownership
		reghdfe `a' `c', absorb(hv001) cluster(hv001)
		//output to spreadsheet
		if `n' == 1 {
			outreg2 using "assets_differences_`c'", replace noni nor2 label excel
		}
		else {
			outreg2 using "assets_differences_`c'", append noni nor2 label excel
		}
	}
}

//generate nicer asset variable labels for graphs
label var hv201_11 "Piped into dwelling"
label var hv201_12 "Piped into yard/plot"
label var hv201_13 "Public tap/standpipe"
label var hv201_21 "Tube well or borehole"
label var hv201_31 "Dug well protected"
label var hv201_32 "Dug well unprotected"
label var hv201_41 "Spring protected"
label var hv201_42 "Spring unprotected"
label var hv201_51 "Rainwater"
label var hv201_61 "Tanker truck"
label var hv201_81 "River/dam/lake/ponds/stream/canal/channel"
label var hv201_91 "Bottled water"
label var hv201_92 "Sachet water"
label var hv201_96 "Other or cart with small tank"
label var hv205_11 "Flush to piped sewer system"
label var hv205_12 "Flush to septic tank"
label var hv205_13 "Flush to pit latrine"
label var hv205_14 "Flush to somewhere else"
label var hv205_15 "Flush to don't know where"
label var hv205_21 "Ventilated improved pit latrine"
label var hv205_22 "Pit latrine with slab"
label var hv205_23 "Pit latrine without slab/open pit"
label var hv205_41 "Bucket toilet"
label var hv205_51 "Hanging toilet/hanging latrine"
label var hv205_61 "No facility/bush/field"
label var hv205_11_sh "Flush to piped sewer system shared"
label var hv205_12_sh "Flush to septic tank shared"
label var hv205_13_sh "Flush to pit latrine shared"
label var hv205_14_sh "Flush to somewhere else shared"
label var hv205_15_sh "Flush to don't know where shared"
label var hv205_21_sh "Ventilated improved pit latrine shared"
label var hv205_22_sh "Pit latrine with slab shared"
label var hv205_23_sh "Pit latrine without slab/open pit shared"
label var hv205_41_sh "Bucket toilet shared"
label var hv205_51_sh "Hanging toilet/hanging latrine shared"
label var hv226_1 "Electricity"
label var hv226_2 "LPG"
label var hv226_3 "Natural gas"
label var hv226_5 "Kerosene"
label var hv226_6 "Coal/lignite"
label var hv226_7 "Charcoal"
label var hv226_8 "Wood"
label var hv226_9 "Straw/shrubs/grass"
label var hv226_10 "Agricultural crop"
label var hv226_95 "No food cooked in house"
label var hv213_11 "Earth/sand"
label var hv213_12 "Dung"
label var hv213_21 "Wood planks"
label var hv213_31 "Parquet/polished wood"
label var hv213_33 "Ceramic/marble/porcelain tiles/terrazo"
label var hv213_34 "Cement"
label var hv213_35 "Woolen carpets/synthetic carpet"
label var hv213_36 "Linoleum/rubber carpet"
label var hv213_96 "Other"
label var hv215_11 "No roof"
label var hv215_12 "Thatch/palm leaf"
label var hv215_21 "Rustic mat"
label var hv215_22 "Palm/bamboo"
label var hv215_23 "Wood planks"
label var hv215_31 "Metal"
label var hv215_32 "Wood"
label var hv215_33 "Calamine/cement fiber"
label var hv215_34 "Ceramic tiles"
label var hv215_35 "Cement"
label var hv215_36 "Roofing shingles"
label var hv215_37 "Asbestos/slate roofing sheets"
label var hv215_96 "Other"
label var hv214_11 "No walls"
label var hv214_12 "Cane/palm/trunks"
label var hv214_13 "Dirt"
label var hv214_21 "Bamboo with mud"
label var hv214_22 "Stone with mud"
label var hv214_23 "Uncovered adobe"
label var hv214_24 "Plywood"
label var hv214_26 "Reused wood"
label var hv214_31 "Cement"
label var hv214_32 "Stone with lime/cement"
label var hv214_33 "Bricks"
label var hv214_34 "Cement blocks"
label var hv214_35 "Covered adobe"
label var hv214_36 "Wood planks/shingles"
label var hv214_96 "Other"

//output to graphs
local categories = "malehoh maleadultpres moremales"
//loop by classification
foreach c of local categories {
	local n = 0
	//loop through assets
	foreach a of local assetlist {
		local label : variable label `a'
		local n = `n' + 1
		drop `c'`n'
		gen `c'`n' = `c'
		label var `c'`n' "`label'"
		replace asset = `a'
		//generate conditional rates of asset ownership
		reghdfe asset `c'`n', absorb(hv001) cluster(hv001)
		est store `a'
	}
	//export graph for source of drinking water
	coefplot hv201_11 hv201_12 hv201_13 hv201_21 hv201_31 hv201_32 hv201_41 hv201_42 hv201_51 hv201_61 hv201_81 hv201_91 hv201_92 hv201_96, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel A: Source of drinking water, size(medium)) pstyle(p1)
	graph export "assets_`c'_A.jpg", as(jpg) quality(100) replace

	//export graph for type of toilet facility
	coefplot hv205_11 hv205_12 hv205_13 hv205_14 hv205_15 hv205_21 hv205_22 hv205_23 hv205_41 hv205_51 hv205_61 hv205_11_sh hv205_12_sh hv205_13_sh hv205_14_sh hv205_15_sh hv205_21_sh hv205_22_sh hv205_23_sh hv205_41_sh hv205_51_sh, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel B: Type of toilet facility, size(medium)) pstyle(p1)
	graph export "assets_`c'_B.jpg", as(jpg) quality(100) replace

	//export graph for type of cooking fuel
	coefplot hv226_1 hv226_2 hv226_3 hv226_5 hv226_6 hv226_7 hv226_8 hv226_9 hv226_10 hv226_95, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel C: Type of cooking fuel, size(medium)) pstyle(p1)
	graph export "assets_`c'_C.jpg", as(jpg) quality(100) replace

	//export graph for main floor material
	coefplot hv213_11 hv213_12 hv213_21 hv213_31 hv213_33 hv213_34 hv213_35 hv213_36 hv213_96, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel D: Main floor material, size(medium)) pstyle(p1)
	graph export "assets_`c'_D.jpg", as(jpg) quality(100) replace

	//export graph for main roof material
	coefplot hv215_11 hv215_12 hv215_21 hv215_22 hv215_23 hv215_31 hv215_32 hv215_33 hv215_34 hv215_35 hv215_36 hv215_37 hv215_96, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel E: Main roof material, size(medium)) pstyle(p1)
	graph export "assets_`c'_E.jpg", as(jpg) quality(100) replace

	//export graph for main wall material
	coefplot hv214_11 hv214_12 hv214_13 hv214_21 hv214_22 hv214_23 hv214_24 hv214_26 hv214_31 hv214_32 hv214_33 hv214_34 hv214_35 hv214_36 hv214_96, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel F: Main wall material, size(medium)) pstyle(p1)
	graph export "assets_`c'_F.jpg", as(jpg) quality(100) replace

	//export graph for other assets
	coefplot hv206 sh110b hv207 sh110d hv208 hv243a hv221 hv209 sh110i sh110j sh110k sh110l sh110m sh110n sh110o sh110p sh110q sh110r sh110s hv243b hv210 hv211 hv243c hv212 hv243d hv247 hv244 memsleep, drop(_cons) xline(0) nokey ylabel(,labsize(small)) graphregion(color(white)) title(Panel G: Other assets, size(medium)) pstyle(p1)
	graph export "assets_`c'_G.jpg", as(jpg) quality(100) replace


}

//save data
rename hv001 dhsclust

cd "${data}"
save "asset_data.dta", replace

/*******************************************************************************
CALCULATE DISTRICT-LEVEL VARIATION IN CLASSIFICATION SYSTEMS
*******************************************************************************/

//This section is automatically turned off because it requires user-generated
//data. To run this section, set global "createmaps" to 1 and ensure you have
//the user-generated data saved in ${data} folder under "GhanaDHS_GADM2.csv"

//To generate CSV needed for this part of the file, follow these steps:
//1) obtain cluster location shapefile from DHS for Ghana 2014
//2) obtain district (GADM2) shapefiles from GADM at gadm.org
//3) load both files in QGIS
//4) join attributes by location, matching districts to cluster that they
//   contain
//5) export joined attributes in csv format to the ${data} folder, named
//   "GhanaDHS_GADM2.csv"

global createmaps = 0

if ${createmaps} == 1 {

	//import file of DHS clusters matched to GADM2 polygons
	cd "${data}"
	import delimited "GhanaDHS_GADM2.csv", clear
	save "district_gadm", replace

	//merge survey data to gadm data for maps
	merge 1:m dhsclust using  "asset_data.dta"
	capture drop _merge

	summ malehoh
	summ maleadultpres
	summ moremales

	//create cluster level averages
	preserve
	bysort dhsclust: egen pct_malehoh = mean(malehoh)
	label var pct_malehoh "Percent of Households with Male Heads"
	bysort dhsclust: egen pct_maleadultpres = mean(maleadultpres)
	label var pct_maleadultpres "Percent of Households with Male Adults in Household"
	bysort dhsclust: egen pct_moremales = mean(moremales)
	label var pct_moremales "Percent of Households with More Male Adults than Female Adults in Household"

	gen tag = 0
	bysort dhsclust: replace tag = 1 if _n == 1
	keep if tag == 1

	summ pct_malehoh
	summ pct_maleadultpres
	summ pct_moremales
	restore

	//create adm2 level (district) averages
	preserve
	bysort gid_2: egen pct_malehoh = mean(malehoh)
	label var pct_malehoh "Percent of Households with Male Heads"
	bysort gid_2: egen pct_maleadultpres = mean(maleadultpres)
	label var pct_maleadultpres "Percent of Households with Male Adults in Household"
	bysort gid_2: egen pct_moremales = mean(moremales)
	label var pct_moremales "Percent of Households with More Male Adults than Female Adults in Household"

	gen tag = 0
	bysort gid_2: replace tag = 1 if _n == 1
	keep if tag == 1
	keep gid_2 pct_maleadultpres pct_malehoh pct_moremales

	//export adm2 level (district) averages for map creation
	save "district_level_gender_classifications", replace

	//the output dataset can be used in QGIS or other software to generate maps
	//showing variation in classification across districts

	summ pct_malehoh
	summ pct_maleadultpres
	summ pct_moremales
	restore

}

/*******************************************************************************
GENERATE WEALTH INDEX ACROSS ALL INDIVIDUALS
*******************************************************************************/

cd "${data}"
use "asset_data.dta", clear

//Wealth index construction across all individuals
//flag obs with missing vlues and replace with mean prior to pca
gen flag_memsleep = 0
gen flag_landarea = 0
replace flag_memsleep = 1 if memsleep == .
replace flag_landarea = 1 if landarea == .
summ memsleep
replace memsleep = r(mean) if memsleep == .
summ landarea
replace landarea = r(mean) if landarea == .

//create no versions of components for all binary variables - later used in addition
foreach a of local assetlist {
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_no =(`a'==0)
	}
}
foreach a of local ruralassetlist {
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		capture gen `a'_no =(`a'==0)
	}
}
foreach a of local urbanassetlist {
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		capture gen `a'_no =(`a'==0)
	}
}

//pca for common components - across all individuals (one component)
pca `assetlist', components(1)
//store means, sds, and components in matrix
matrix means_all = e(means)
matrix sds_all = e(sds)
matrix components_all = e(L)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_all = 0
local count = 0
foreach a of local assetlist {
	local count = `count' +1
	if "`a'" != "memsleep" {
		gen `a'_score = ((((1-means_all[1,`count'])/sds_all[1,`count'])*components_all[`count',1])*`a') + ((((0-means_all[1,`count'])/sds_all[1,`count'])*components_all[`count',1])*`a'_no)
		replace index_all = index_all + `a'_score
	}
	if "`a'" == "memsleep" {
		gen `a'_score = ((`a'/sds_all[1,`count'])*components_all[`count',1])
		replace index_all = index_all + `a'_score
	}
}

//pca for rural components - across all rural individuals (one component)
pca `ruralassetlist' if hv025==2, components(1)
//store means, sds, and components in matrix
matrix means_rural_all = e(means)
matrix sds_rural_all = e(sds)
matrix components_rural_all = e(L)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_rural_all = 0
local count = 0
foreach a of local ruralassetlist {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		display "`a' `count'"
		gen `a'_rural_score = ((((1-means_rural_all[1,`count'])/sds_rural_all[1,`count'])*components_rural_all[`count',1])*`a') + ((((0-means_rural_all[1,`count'])/sds_rural_all[1,`count'])*components_rural_all[`count',1])*`a'_no)
		replace index_rural_all = index_rural_all + `a'_rural_score
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_rural_score = ((`a'/sds_rural_all[1,`count'])*components_rural_all[`count',1])
		replace index_rural_all = index_rural_all + `a'_rural_score
	}
}
replace index_rural_all = . if hv025!=2

//pca for urban components - across all urban individuals
pca `urbanassetlist' if hv025==1, components(1)
//store means, sds, and components in matrix
matrix means_urban_all = e(means)
matrix sds_urban_all = e(sds)
matrix components_urban_all = e(L)

//redefine urban component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv213_96 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
local dropped = "hv213_96 hv246i_2"
local urbanassetlist2 = "`urbanassetlist'"
foreach d of local dropped {
	local urbanassetlist2 = subinstr("`urbanassetlist2'", "`d'", "", 1)
}
local urbanassetlist2 = subinstr("`urbanassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_urban_all = 0
local count = 0
foreach a of local urbanassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_urban_score = ((((1-means_urban_all[1,`count'])/sds_urban_all[1,`count'])*components_urban_all[`count',1])*`a') + ((((0-means_urban_all[1,`count'])/sds_urban_all[1,`count'])*components_urban_all[`count',1])*`a'_no)
		replace index_urban_all = index_urban_all + `a'_urban_score
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_urban_score = ((`a'/sds_urban_all[1,`count'])*components_urban_all[`count',1])
		replace index_urban_all = index_urban_all + `a'_urban_score
	}
}
replace index_urban_all = . if hv025!=1

//generate final wealth index using common and rural and urban components
gen wealthindex_all = .

//find weights for common vs rural components
reg index_all index_rural_all
local cons_rural = e(b)[1,2]
local beta_rural = e(b)[1,1]

replace wealthindex_all = `cons_rural' + `beta_rural'*index_rural_all if hv025 == 2

//find weights for common vs urban components
reg index_all index_urban_all
local cons_urban = e(b)[1,2]
local beta_urban = e(b)[1,1]

replace wealthindex_all = `cons_urban' + `beta_urban'*index_urban_all if hv025 == 1

//calculate weighted number of household members
gen hhmemwt = hv012*hv005/1000000
replace hhmemwt = hv013*hv005/1000000
label var hhmemwt "HH members weighting for Index"

//calculate quintiles
xtile wealthindex_all_q = wealthindex_all [pweight=hhmemwt], n(5)
label var hv270 "DHS wealth index"
label var wealthindex_all_q "Our wealth index (all)"

//visually compare our index with the DHS WI
tab wealthindex_all_q hv270

//look at symmetry in off-diagonals
count if hv270 < wealthindex_all_q
count if hv270 > wealthindex_all_q
count if hv270 == wealthindex_all_q

/*******************************************************************************
GENERATE WEALTH INDEX BASED ONLY GENDER OF HOUSEHOLD HEAD
*******************************************************************************/

//reinstate missing values for continuous variables
replace memsleep = . if flag_memsleep == 1
replace landarea = . if flag_landarea == 1
//replace missing values for continuous variables to average for female headed hhs
summ memsleep if malehoh == 0
replace memsleep = r(mean) if memsleep == . & malehoh == 0
summ landarea if malehoh == 0
replace landarea = r(mean) if landarea == . & malehoh == 0
//replace missing values for continuous variables to average for male headed hhs
summ memsleep if malehoh == 1
replace memsleep = r(mean) if memsleep == . & malehoh == 1
summ landarea if malehoh == 1
replace landarea = r(mean) if landarea == . & malehoh == 1

//wealth index for female hohs
//pca for common components - female hohs (one component)
pca `assetlist' if malehoh == 0, components(1)
//store means, sds, and components in matrix
matrix means_fhoh = e(means)
matrix sds_fhoh = e(sds)
matrix components_fhoh = e(L)

//redefine common component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv213_96 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
local dropped = "hv213_96 hv215_96"
local assetlist2 = "`assetlist'"
foreach d of local dropped {
	local assetlist2 = subinstr("`assetlist2'", "`d'", "", 1)
}
local assetlist2 = subinstr("`assetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_fhoh = 0
local count = 0
foreach a of local assetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" {
		gen `a'_score_fhoh = ((((1-means_fhoh[1,`count'])/sds_fhoh[1,`count'])*components_fhoh[`count',1])*`a') + ((((0-means_fhoh[1,`count'])/sds_fhoh[1,`count'])*components_fhoh[`count',1])*`a'_no)
		replace index_fhoh = index_fhoh + `a'_score_fhoh
	}
	if "`a'" == "memsleep" {
		gen `a'_score_fhoh = ((`a'/sds_fhoh[1,`count'])*components_fhoh[`count',1])
		replace index_fhoh = index_fhoh + `a'_score_fhoh
	}
}
replace index_fhoh = . if malehoh == 1

//pca for rural components - across rural female hohs (one component)
pca `ruralassetlist' if hv025==2 & malehoh == 0, components(1)
//store means, sds, and components in matrix
matrix means_rural_fhoh = e(means)
matrix sds_rural_fhoh = e(sds)
matrix components_rural_fhoh = e(L)

//redefine rural component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv201_96 dropped because of zero variance)
//(hv226_1 dropped because of zero variance)
//(hv226_5 dropped because of zero variance)
//(hv213_31 dropped because of zero variance)
//(hv213_96 dropped because of zero variance)
//(hv215_32 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
//(hv246b_2 dropped because of zero variance)
//(hv246h_3 dropped because of zero variance)
//(hv246i_0 dropped because of zero variance)
//(hv246i_1 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
//(hv246i_3 dropped because of zero variance)
//(hv246k_3 dropped because of zero variance)
local dropped = "hv201_96 hv226_1 hv226_5 hv213_31 hv213_96 hv215_32 hv215_96 hv246b_2 hv246h_3 hv246i_0 hv246i_1 hv246i_2 hv246i_3 hv246k_3"
local ruralassetlist2 = "`ruralassetlist'"
foreach d of local dropped {
	local ruralassetlist2 = subinstr("`ruralassetlist2'", "`d'", "", 1)
}
local ruralassetlist2 = subinstr("`ruralassetlist2'", "  ", " ", .)
display "`ruralassetlist2'"

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_rural_fhoh = 0
local count = 0
foreach a of local ruralassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		display "`a' `count'"
		gen `a'_rural_score_fhoh = ((((1-means_rural_fhoh[1,`count'])/sds_rural_fhoh[1,`count'])*components_rural_fhoh[`count',1])*`a') + ((((0-means_rural_fhoh[1,`count'])/sds_rural_fhoh[1,`count'])*components_rural_fhoh[`count',1])*`a'_no)
		replace index_rural_fhoh = index_rural_fhoh + `a'_rural_score_fhoh
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_rural_score_fhoh = ((`a'/sds_rural_fhoh[1,`count'])*components_rural_fhoh[`count',1])
		replace index_rural_fhoh = index_rural_fhoh + `a'_rural_score_fhoh
	}
}
replace index_rural_fhoh = . if hv025!=2 | malehoh == 1

//pca for urban components - across urban female hohs
pca `urbanassetlist' if hv025==1 & malehoh == 0, components(1)
//store means, sds, and components in matrix
matrix means_urban_fhoh = e(means)
matrix sds_urban_fhoh = e(sds)
matrix components_urban_fhoh = e(L)

//redefine urban component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv226_6 dropped because of zero variance)
//(hv213_12 dropped because of zero variance)
//(hv213_96 dropped because of zero variance)
//(hv215_21 dropped because of zero variance)
//(hv215_22 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
//(hv246b_2 dropped because of zero variance)
//(hv246b_3 dropped because of zero variance)
//(hv246i_1 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
//(hv246j_3 dropped because of zero variance)
local dropped = "hv226_6 hv213_12 hv213_96 hv215_21 hv215_22 hv215_96 hv246b_2 hv246b_3 hv246i_1 hv246i_2 hv246j_3"
local urbanassetlist2 = "`urbanassetlist'"
foreach d of local dropped {
	local urbanassetlist2 = subinstr("`urbanassetlist2'", "`d'", "", 1)
}
local urbanassetlist2 = subinstr("`urbanassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_urban_fhoh = 0
local count = 0
foreach a of local urbanassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_urban_score_fhoh = ((((1-means_urban_fhoh[1,`count'])/sds_urban_fhoh[1,`count'])*components_urban_fhoh[`count',1])*`a') + ((((0-means_urban_fhoh[1,`count'])/sds_urban_fhoh[1,`count'])*components_urban_fhoh[`count',1])*`a'_no)
		replace index_urban_fhoh = index_urban_fhoh + `a'_urban_score_fhoh
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_urban_score_fhoh = ((`a'/sds_urban_fhoh[1,`count'])*components_urban_fhoh[`count',1])
		replace index_urban_fhoh = index_urban_fhoh + `a'_urban_score_fhoh
	}
}
replace index_urban_fhoh = . if hv025!=1 | malehoh == 1

//generate final wealth index using common and rural and urban components
gen wealthindex_fhoh = .

//find weights for common vs rural components
reg index_fhoh index_rural_fhoh
local cons_rural_fhoh = e(b)[1,2]
local beta_rural_fhoh = e(b)[1,1]

replace wealthindex_fhoh = `cons_rural_fhoh' + `beta_rural_fhoh'*index_rural_fhoh if hv025 == 2 & malehoh == 0

//find weights for common vs urban components
reg index_fhoh index_urban_fhoh
local cons_urban_fhoh = e(b)[1,2]
local beta_urban_fhoh = e(b)[1,1]

replace wealthindex_fhoh = `cons_urban_fhoh' + `beta_urban_fhoh'*index_urban_fhoh if hv025 == 1 & malehoh == 0

//calculate quintiles
xtile wealthindex_fhoh_q = wealthindex_fhoh [pweight=hhmemwt], n(5)
label var wealthindex_fhoh_q "Our wealth index (female HoHs)"

//visually compare our index with the DHS WI
tab wealthindex_all_q wealthindex_fhoh_q

//wealth index for male hohs
//pca for common components - male hohs (one component)
pca `assetlist' if malehoh == 1, components(1)
//store means, sds, and components in matrix
matrix means_mhoh = e(means)
matrix sds_mhoh = e(sds)
matrix components_mhoh = e(L)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_mhoh = 0
local count = 0
foreach a of local assetlist {
	local count = `count' +1
	if "`a'" != "memsleep" {
		gen `a'_score_mhoh = ((((1-means_mhoh[1,`count'])/sds_mhoh[1,`count'])*components_mhoh[`count',1])*`a') + ((((0-means_mhoh[1,`count'])/sds_mhoh[1,`count'])*components_mhoh[`count',1])*`a'_no)
		replace index_mhoh = index_mhoh + `a'_score_mhoh
	}
	if "`a'" == "memsleep" {
		gen `a'_score_mhoh = ((`a'/sds_mhoh[1,`count'])*components_mhoh[`count',1])
		replace index_mhoh = index_mhoh + `a'_score_mhoh
	}
}
replace index_mhoh = . if malehoh == 0

//pca for rural components - across rural male hohs (one component)
pca `ruralassetlist' if hv025==2 & malehoh == 1, components(1)
//store means, sds, and components in matrix
matrix means_rural_mhoh = e(means)
matrix sds_rural_mhoh = e(sds)
matrix components_rural_mhoh = e(L)

//redefine rural component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv205_14 dropped because of zero variance)
//(hv205_41 dropped because of zero variance)
//(hv205_15_sh dropped because of zero variance)
//(hv205_41_sh dropped because of zero variance)
//(hv215_33 dropped because of zero variance)
local dropped = "hv205_14 hv205_41 hv205_15_sh hv205_41_sh hv215_33"
local ruralassetlist2 = "`ruralassetlist'"
foreach d of local dropped {
	local ruralassetlist2 = subinstr("`ruralassetlist2'", "`d'", "", 1)
}
local ruralassetlist2 = subinstr("`ruralassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_rural_mhoh = 0
local count = 0
foreach a of local ruralassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		display "`a' `count'"
		gen `a'_rural_score_mhoh = ((((1-means_rural_mhoh[1,`count'])/sds_rural_mhoh[1,`count'])*components_rural_mhoh[`count',1])*`a') + ((((0-means_rural_mhoh[1,`count'])/sds_rural_mhoh[1,`count'])*components_rural_mhoh[`count',1])*`a'_no)
		replace index_rural_mhoh = index_rural_mhoh + `a'_rural_score_mhoh
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_rural_score_mhoh = ((`a'/sds_rural_mhoh[1,`count'])*components_rural_mhoh[`count',1])
		replace index_rural_mhoh = index_rural_mhoh + `a'_rural_score_mhoh
	}
}
replace index_rural_mhoh = . if hv025!=2 | malehoh == 0

//pca for urban components - across urban male hohs
pca `urbanassetlist' if hv025==1 & malehoh == 1, components(1)
//store means, sds, and components in matrix
matrix means_urban_mhoh = e(means)
matrix sds_urban_mhoh = e(sds)
matrix components_urban_mhoh = e(L)

//redefine urban component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv213_96 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
local dropped = "hv213_96 hv246i_2"
local urbanassetlist2 = "`urbanassetlist'"
foreach d of local dropped {
	local urbanassetlist2 = subinstr("`urbanassetlist2'", "`d'", "", 1)
}
local urbanassetlist2 = subinstr("`urbanassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_urban_mhoh = 0
local count = 0
foreach a of local urbanassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_urban_score_mhoh = ((((1-means_urban_mhoh[1,`count'])/sds_urban_mhoh[1,`count'])*components_urban_mhoh[`count',1])*`a') + ((((0-means_urban_mhoh[1,`count'])/sds_urban_mhoh[1,`count'])*components_urban_mhoh[`count',1])*`a'_no)
		replace index_urban_mhoh = index_urban_mhoh + `a'_urban_score_mhoh
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_urban_score_mhoh = ((`a'/sds_urban_mhoh[1,`count'])*components_urban_mhoh[`count',1])
		replace index_urban_mhoh = index_urban_mhoh + `a'_urban_score_mhoh
	}
}
replace index_urban_mhoh = . if hv025!=1 | malehoh == 0

//generate final wealth index using common and rural and urban components
gen wealthindex_mhoh = .

//find weights for common vs rural components
reg index_mhoh index_rural_mhoh
local cons_rural_mhoh = e(b)[1,2]
local beta_rural_mhoh = e(b)[1,1]

replace wealthindex_mhoh = `cons_rural_mhoh' + `beta_rural_mhoh'*index_rural_mhoh if hv025 == 2 & malehoh == 1

//find weights for common vs urban components
reg index_mhoh index_urban_mhoh
local cons_urban_mhoh = e(b)[1,2]
local beta_urban_mhoh = e(b)[1,1]

replace wealthindex_mhoh = `cons_urban_mhoh' + `beta_urban_mhoh'*index_urban_mhoh if hv025 == 1 & malehoh == 1

//calculate quintiles
xtile wealthindex_mhoh_q = wealthindex_mhoh [pweight=hhmemwt], n(5)
label var wealthindex_mhoh_q "Our wealth index (male HoHs)"

//visually compare our index with the DHS WI
tab wealthindex_all_q wealthindex_mhoh_q

/*******************************************************************************
GENERATE WEALTH INDEX BASED ONLY ON PRESENCE OF AN ADULT PERMANENT MALE
*******************************************************************************/
//reinstate missing values for continuous variables
replace memsleep = . if flag_memsleep == 1
replace landarea = . if flag_landarea == 1
//replace missing values for continuous variables to average for female headed hhs
summ memsleep if maleadultpres == 0
replace memsleep = r(mean) if memsleep == . & maleadultpres == 0
summ landarea if maleadultpres == 0
replace landarea = r(mean) if landarea == . & maleadultpres == 0
//replace missing values for continuous variables to average for male headed hhs
summ memsleep if maleadultpres == 1
replace memsleep = r(mean) if memsleep == . & maleadultpres == 1
summ landarea if maleadultpres == 1
replace landarea = r(mean) if landarea == . & maleadultpres == 1

//wealth index for hhs with only females present
//pca for common components - females only (one component)
pca `assetlist' if maleadultpres == 0, components(1)
//store means, sds, and components in matrix
matrix means_fpres = e(means)
matrix sds_fpres = e(sds)
matrix components_fpres = e(L)

//redefine common component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv213_96 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
local dropped = "hv213_96 hv215_96"
local assetlist2 = "`assetlist'"
foreach d of local dropped {
	local assetlist2 = subinstr("`assetlist2'", "`d'", "", 1)
}
local assetlist2 = subinstr("`assetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_fpres = 0
local count = 0
foreach a of local assetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" {
		gen `a'_score_fpres = ((((1-means_fpres[1,`count'])/sds_fpres[1,`count'])*components_fpres[`count',1])*`a') + ((((0-means_fpres[1,`count'])/sds_fpres[1,`count'])*components_fpres[`count',1])*`a'_no)
		replace index_fpres = index_fpres + `a'_score_fpres
	}
	if "`a'" == "memsleep" {
		gen `a'_score_fpres = ((`a'/sds_fpres[1,`count'])*components_fpres[`count',1])
		replace index_fpres = index_fpres + `a'_score_fpres
	}
}
replace index_fpres = . if maleadultpres == 1

//pca for rural components - across rural hhs with only female (one component)
pca `ruralassetlist' if hv025==2 & maleadultpres == 0, components(1)
//store means, sds, and components in matrix
matrix means_rural_fpres = e(means)
matrix sds_rural_fpres = e(sds)
matrix components_rural_fpres = e(L)

//redefine rural component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv201_61 dropped because of zero variance)
//(hv201_96 dropped because of zero variance)
//(hv226_1 dropped because of zero variance)
//(hv226_5 dropped because of zero variance)
//(hv213_31 dropped because of zero variance)
//(hv213_96 dropped because of zero variance)
//(hv215_23 dropped because of zero variance)
//(hv215_32 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
//(hv246b_2 dropped because of zero variance)
//(hv246c_2 dropped because of zero variance)
//(hv246h_2 dropped because of zero variance)
//(hv246h_3 dropped because of zero variance)
//(hv246i_0 dropped because of zero variance)
//(hv246i_1 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
//(hv246i_3 dropped because of zero variance)
//(hv246j_3 dropped because of zero variance)
//(hv246k_3 dropped because of zero variance)
local dropped = "hv201_61 hv201_96 hv226_1 hv226_5 hv213_31 hv213_96 hv215_23 hv215_32 hv215_96 hv246b_2 hv246c_2 hv246h_2 hv246h_3 hv246i_0 hv246i_1 hv246i_2 hv246i_3 hv246j_3 hv246k_3"
local ruralassetlist2 = "`ruralassetlist'"
foreach d of local dropped {
	local ruralassetlist2 = subinstr("`ruralassetlist2'", "`d'", "", 1)
}
local ruralassetlist2 = subinstr("`ruralassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_rural_fpres = 0
local count = 0
foreach a of local ruralassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		display "`a' `count'"
		gen `a'_rural_score_fpres = ((((1-means_rural_fpres[1,`count'])/sds_rural_fpres[1,`count'])*components_rural_fpres[`count',1])*`a') + ((((0-means_rural_fpres[1,`count'])/sds_rural_fpres[1,`count'])*components_rural_fpres[`count',1])*`a'_no)
		replace index_rural_fpres = index_rural_fpres + `a'_rural_score_fpres
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_rural_score_fpres = ((`a'/sds_rural_fpres[1,`count'])*components_rural_fpres[`count',1])
		replace index_rural_fpres = index_rural_fpres + `a'_rural_score_fpres
	}
}
replace index_rural_fpres = . if hv025!=2 | maleadultpres == 1

//pca for urban components - across urban hhs with only females
pca `urbanassetlist' if hv025==1 & maleadultpres == 0, components(1)
//store means, sds, and components in matrix
matrix means_urban_fpres = e(means)
matrix sds_urban_fpres = e(sds)
matrix components_urban_fpres = e(L)

//redefine urban component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv226_6 dropped because of zero variance)
//(hv213_12 dropped because of zero variance)
//(hv213_96 dropped because of zero variance)
//(hv215_21 dropped because of zero variance)
//(hv215_22 dropped because of zero variance)
//(hv215_96 dropped because of zero variance)
//(hv246b_2 dropped because of zero variance)
//(hv246b_3 dropped because of zero variance)
//(hv246c_1 dropped because of zero variance)
//(hv246i_0 dropped because of zero variance)
//(hv246i_1 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
//(hv246i_3 dropped because of zero variance)
//(hv246j_3 dropped because of zero variance)
//(hv246k_2 dropped because of zero variance)
local dropped = "hv226_6 hv213_12 hv213_96 hv215_21 hv215_22 hv215_96 hv246b_2 hv246b_3 hv246c_1 hv246i_0 hv246i_1 hv246i_2 hv246i_3 hv246j_3 hv246k_2"
local urbanassetlist2 = "`urbanassetlist'"
foreach d of local dropped {
	local urbanassetlist2 = subinstr("`urbanassetlist2'", "`d'", "", 1)
}
local urbanassetlist2 = subinstr("`urbanassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_urban_fpres = 0
local count = 0
foreach a of local urbanassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_urban_score_fpres = ((((1-means_urban_fpres[1,`count'])/sds_urban_fpres[1,`count'])*components_urban_fpres[`count',1])*`a') + ((((0-means_urban_fpres[1,`count'])/sds_urban_fpres[1,`count'])*components_urban_fpres[`count',1])*`a'_no)
		replace index_urban_fpres = index_urban_fpres + `a'_urban_score_fpres
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_urban_score_fpres = ((`a'/sds_urban_fpres[1,`count'])*components_urban_fpres[`count',1])
		replace index_urban_fpres = index_urban_fpres + `a'_urban_score_fpres
	}
}
replace index_urban_fpres = . if hv025!=1 | maleadultpres == 1

//generate final wealth index using common and rural and urban components
gen wealthindex_fpres = .

//find weights for common vs rural components
reg index_fpres index_rural_fpres
local cons_rural_fpres = e(b)[1,2]
local beta_rural_fpres = e(b)[1,1]

replace wealthindex_fpres = `cons_rural_fpres' + `beta_rural_fpres'*index_rural_fpres if hv025 == 2 & maleadultpres == 0

//find weights for common vs urban components
reg index_fpres index_urban_fpres
local cons_urban_fpres = e(b)[1,2]
local beta_urban_fpres = e(b)[1,1]

replace wealthindex_fpres = `cons_urban_fpres' + `beta_urban_fpres'*index_urban_fpres if hv025 == 1 & maleadultpres == 0

//calculate quintiles
xtile wealthindex_fpres_q = wealthindex_fpres [pweight=hhmemwt], n(5)
label var wealthindex_fpres_q "Our wealth index (HH w/ female adults only)"

//visually compare our index with the DHS WI
tab wealthindex_all_q wealthindex_fpres_q

//wealth index for hhs with male adult present
//pca for common components - hhs with male adult present (one component)
pca `assetlist' if maleadultpres == 1, components(1)
//store means, sds, and components in matrix
matrix means_mpres = e(means)
matrix sds_mpres = e(sds)
matrix components_mpres = e(L)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_mpres = 0
local count = 0
foreach a of local assetlist {
	local count = `count' +1
	if "`a'" != "memsleep" {
		gen `a'_score_mpres = ((((1-means_mpres[1,`count'])/sds_mpres[1,`count'])*components_mpres[`count',1])*`a') + ((((0-means_mpres[1,`count'])/sds_mpres[1,`count'])*components_mpres[`count',1])*`a'_no)
		replace index_mpres = index_mpres + `a'_score_mpres
	}
	if "`a'" == "memsleep" {
		gen `a'_score_mpres = ((`a'/sds_mpres[1,`count'])*components_mpres[`count',1])
		replace index_mpres = index_mpres + `a'_score_mpres
	}
}
replace index_mpres = . if maleadultpres == 0

//pca for rural components - across rural hhs with male present (one component)
pca `ruralassetlist' if hv025==2 & maleadultpres == 1, components(1)
//store means, sds, and components in matrix
matrix means_rural_mpres = e(means)
matrix sds_rural_mpres = e(sds)
matrix components_rural_mpres = e(L)

//redefine rural component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv205_14 dropped because of zero variance)
//(hv205_41 dropped because of zero variance)
//(hv205_15_sh dropped because of zero variance)
//(hv205_41_sh dropped because of zero variance)
//(hv215_33 dropped because of zero variance)
local dropped = "hv205_14 hv205_41 hv205_15_sh hv205_41_sh hv215_33"
local ruralassetlist2 = "`ruralassetlist'"
foreach d of local dropped {
	local ruralassetlist2 = subinstr("`ruralassetlist2'", "`d'", "", 1)
}
local ruralassetlist2 = subinstr("`ruralassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_rural_mpres = 0
local count = 0
foreach a of local ruralassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		display "`a' `count'"
		gen `a'_rural_score_mpres = ((((1-means_rural_mpres[1,`count'])/sds_rural_mpres[1,`count'])*components_rural_mpres[`count',1])*`a') + ((((0-means_rural_mpres[1,`count'])/sds_rural_mpres[1,`count'])*components_rural_mpres[`count',1])*`a'_no)
		replace index_rural_mpres = index_rural_mpres + `a'_rural_score_mpres
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_rural_score_mpres = ((`a'/sds_rural_mpres[1,`count'])*components_rural_mpres[`count',1])
		replace index_rural_mpres = index_rural_mpres + `a'_rural_score_mpres
	}
}
replace index_rural_mpres = . if hv025!=2 | maleadultpres == 0

//pca for urban components - across urban male hohs
pca `urbanassetlist' if hv025==1 & maleadultpres == 1, components(1)
//store means, sds, and components in matrix
matrix means_urban_mpres = e(means)
matrix sds_urban_mpres = e(sds)
matrix components_urban_mpres = e(L)

//redefine urban component list, dropping vars with zero variance (naturally drop out if try to include them in pca)
//(hv213_96 dropped because of zero variance)
//(hv246i_2 dropped because of zero variance)
local dropped = "hv213_96 hv246i_2"
local urbanassetlist2 = "`urbanassetlist'"
foreach d of local dropped {
	local urbanassetlist2 = subinstr("`urbanassetlist2'", "`d'", "", 1)
}
local urbanassetlist2 = subinstr("`urbanassetlist2'", "  ", " ", .)

//loop through all components and create a component score for each obs
//if they own a binary component, then it is equal to ((1-mean)/sd)*componentscore
//if they don't own a binary component, then it is equal to ((0-mean)/sd)*componentscore
//for continuous variables, it is equal to (component/sd)*componentscore
gen index_urban_mpres = 0
local count = 0
foreach a of local urbanassetlist2 {
	local count = `count' +1
	if "`a'" != "memsleep" & "`a'" != "landarea" {
		gen `a'_urban_score_mpres = ((((1-means_urban_mpres[1,`count'])/sds_urban_mpres[1,`count'])*components_urban_mpres[`count',1])*`a') + ((((0-means_urban_mpres[1,`count'])/sds_urban_mpres[1,`count'])*components_urban_mpres[`count',1])*`a'_no)
		replace index_urban_mpres = index_urban_mpres + `a'_urban_score_mpres
	}
	if "`a'" == "memsleep" | "`a'" == "landarea" {
		gen `a'_urban_score_mpres = ((`a'/sds_urban_mpres[1,`count'])*components_urban_mpres[`count',1])
		replace index_urban_mpres = index_urban_mpres + `a'_urban_score_mpres
	}
}
replace index_urban_mpres = . if hv025!=1 | maleadultpres == 0

//generate final wealth index using common and rural and urban components
gen wealthindex_mpres = .

//find weights for common vs rural components
reg index_mpres index_rural_mpres
local cons_rural_mpres = e(b)[1,2]
local beta_rural_mpres = e(b)[1,1]

replace wealthindex_mpres = `cons_rural_mpres' + `beta_rural_mpres'*index_rural_mpres if hv025 == 2 & maleadultpres == 1

//find weights for common vs urban components
reg index_mpres index_urban_mpres
local cons_urban_mpres = e(b)[1,2]
local beta_urban_mpres = e(b)[1,1]

replace wealthindex_mpres = `cons_urban_mpres' + `beta_urban_mpres'*index_urban_mpres if hv025 == 1 & maleadultpres == 1

//calculate quintiles
xtile wealthindex_mpres_q = wealthindex_mpres [pweight=hhmemwt], n(5)
label var wealthindex_mpres_q "Our wealth index (HHs with 1+ males)"

//visually compare our index with the DHS WI
tab wealthindex_all_q wealthindex_mpres_q

//close out do-file
log close
