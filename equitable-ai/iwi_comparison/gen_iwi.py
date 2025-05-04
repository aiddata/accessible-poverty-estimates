"""
Generate the International Wealth Index (IWI) for Ghana 2014 DHS data.
"""
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# USER VARIABLES

base_path = Path("/home/userx/Desktop/accessible-poverty-estimates/equitable-ai")

household_data_path = base_path / "GH_2014_DHS/GHHR72DT/GHHR72FL.DTA"

output_dir = base_path / 'iwi_comparison'

output_path = output_dir / 'ghana_2014_iwi.csv'


# -----------------------------------------------------------------------------
# SURVEY VARIABLES

iwi_constant = 25.00447


def memsleep(x):
    rooms = x["hv216"] if x["hv216"] > 0 else 1
    if x["hv012"] > 0:
        return x["hv012"] / rooms
    else:
        return x["hv013"] / rooms


cheap_utensils_dict = {
    "sh110p": "Bed",
    "sh110q": "Table",
    "sh110r": "Cabinet/cupboard",
    "hv243b": "Watch",
    "sh110b": "Wall clock",
    "hv207": "Radio",
    "hv210": "Bicycle",
    "hv243c": "Animal-drawn cart",
    "sh110m": "Photo camera",
    "sh110o": "Sewing machine"
}

expensive_utensils_dict = {
    "sh110l": "Computer/tablet computer",
    "hv206": "Electricity",
    "sh110d": "Black/white television",
    "hv208": "Color television",
    "hv243a": "Mobile telephone",
    "hv221": "Telephone (non-mobile)",
    "hv209": "Refrigerator",
    "sh110i": "Freezer",
    "sh110j": "Electric generator/inverter",
    "sh110k": "Washing machine",
    "hv211": "Motorcycle/scooter",
    "hv212": "Car/truck",
    "hv243d": "Boat with a motor"
}

asset_dicts = [
    {
        'label': 'Television',
        'weight': 8.612657,
        'func': lambda x: max(x["sh110d"], x["hv208"])
    },
    {
        'label': 'Refrigerator',
        'weight': 8.429076,
        'func': lambda x: x["hv209"]
    },
    {
        'label': 'Phone',
        'weight': 7.127699,
        'func': lambda x: max(x["hv243a"], x["hv221"])
    },
    {
        'label': 'Car',
        'weight': 4.651382,
        'func': lambda x: x["hv212"]
    },
    {
        'label': 'Bike',
        'weight': 1.84686,
        'func': lambda x: x["hv210"]
    },
    {
        'label': 'Access to electricity',
        'weight': 8.056664,
        'func': lambda x: max(x["hv206"], x["sh110j"])
    },
    {
        'label': 'Low quality Water source',
        'weight': -6.306477,
        'func': lambda x: x["hv201"] in [32, 42, 43, 81, 62, 96, 51]
    },
    {
        'label': 'Medium quality Water source',
        'weight': -2.302023,
        'func': lambda x: x["hv201"] in [13, 21, 31, 41, 61]
    },
    {
        'label': 'High quality Water source',
        'weight': 7.952443,
        'func': lambda x: x["hv201"] in [11, 12, 71, 91, 72, 92]
    },
    {
        'label': 'Low quality Floor material',
        'weight': -7.558471,
        'func': lambda x: x["hv213"] in [11, 12]
    },
    {
        'label': 'Medium quality Floor material',
        'weight': 1.227531,
        'func': lambda x: x["hv213"] in [21, 22]
    },
    {
        'label': 'High quality Floor material',
        'weight': 6.107428,
        'func': lambda x: x["hv213"] in [31, 32, 33, 34, 35, 96]
    },
    {
        'label': 'Low quality Toilet facility',
        'weight': -7.439841,
        'func': lambda x: x["hv205"] in [23, 42, 43, 51, 61]
    },
    {
        'label': 'Medium quality Toilet facility',
        'weight': -1.090393,
        'func': lambda x: x["hv205"] in [21, 22, 31] or (x["hv225"] == 1 and x["hv205"] not in [23, 42, 43, 51, 31, 61])
    },
    {
        'label': 'High quality Toilet facility',
        'weight': 8.140637,
        'func': lambda x: x["hv205"] in [11, 12, 13, 14, 15] and x["hv225"] != 1
    },
    {
        'label': 'Zero or one sleeping rooms',
        'weight': -3.699681,
        'func': lambda x: x['memsleep'] < 2
    },
    {
        'label': 'Two sleeping rooms',
        'weight': 0.38405,
        'func': lambda x: x['memsleep'] == 2
    },
    {
        'label': 'Three or more sleeping rooms',
        'weight': 3.445009,
        'func': lambda x: x['memsleep'] > 2
    },
    {
        'label': 'Expensive utensils',
        'weight': 6.507283,
        'func': lambda x: any([x[i] for i in expensive_utensils_dict.keys()])
    },
    {
        'label': 'Cheap utensils',
        'weight': 4.118394,
        'func': lambda x: any([x[i] for i in cheap_utensils_dict.keys()])
    }
]


# -----------------------------------------------------------------------------
# SURVEY CODE

hr_reader = pd.read_stata(household_data_path, convert_categoricals=False, iterator=True)
hr_dict = hr_reader.variable_labels()

with hr_reader:
    hr_df = hr_reader.read()
    # hr_df.rename(columns=hr_dict, inplace=True)

hr_df['hhid'] = hr_df['hhid'].apply(lambda x: x.replace(' ', ''))

hr_df['memsleep'] = hr_df.apply(memsleep, axis=1)


df = hr_df[['hhid', 'hv271', 'hv219']].copy()
df.rename(columns={
    'hv271': 'dhs_wi',
    'hv219': 'gender'
}, inplace=True)

df['gender'] = df.gender.apply(lambda x: ['male', 'female'][x-1])


# -----------------------------------------------------------------------------
# IWI CODE

def expensive_utensil(x):
    # any items over $250 USD value
    #   examples: washer, dryer, computer, motorbike, motorboat, air conditioner, or generator
    if x["Expensive utensils"] == 1:
        return 1
    # assumed based on ownership of car
    if x["Car"] == 1:
        return 1
    else:
        return 0


def cheap_utensil(x):
    # any items under $50 USD value
    #   examples: chair, table, clock, watch, water cooker, radio, fan or mixer
    if x["Cheap utensils"] == 1:
        return 1
    # assumed based on ownership of expensive utensils
    elif x["Expensive utensils"] == 1:
        return 1
    # assumed based on ownership of high quality floor or toilet facility
    elif x["High quality Floor material"] == 1 or x["High quality Toilet facility"] == 1:
        return 1
    # assumed based on ownership of TV, fridge, phone, bicycle, car
    elif x["Television"] == 1 or x["Refrigerator"] == 1 or x["Phone"] == 1 or x["Bike"] == 1 or x["Car"] == 1:
        return 1
    else:
        return 0


for v in asset_dicts:
    print(v['label'])
    if v['label'] in df.columns.to_list():
        continue
    try:
        print("\t...applying function")
        data = hr_df.apply(v["func"], axis=1)
        df[v["label"]] = data
    except Exception as e:
        print(v)
        raise

df.fillna(0, inplace=True)
for c in df.columns:
    if c not in ['hhid', 'gender', 'dhs_wi']:
        df[c] = df[c].astype(int)

df['Expensive utensils'] = df.apply(expensive_utensil, axis=1)
df['Cheap utensils'] = df.apply(cheap_utensil, axis=1)


# IWI = constant + Σ(Wn ⋅ Xn)

df['iwi'] = iwi_constant + df.apply(lambda x: sum([v['weight'] * x[v['label']] for v in asset_dicts]), axis=1)

output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
