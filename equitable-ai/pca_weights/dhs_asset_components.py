asset_dict = {
    "hv206": {
        "label": "Electricity",
        "func": lambda x: x["hv206"]
    },
    "sh110b": {
        "label": "Wall clock",
        "func": lambda x: x["sh110b"]
    },
    "hv207": {
        "label": "Radio",
        "func": lambda x: x["hv207"]
    },
    "sh110d": {
        "label": "Black/white television",
        "func": lambda x: x["sh110d"]
    },
    "hv208": {
        "label": "Color television",
        "func": lambda x: x["hv208"]
    },
    "hv243a": {
        "label": "Mobile telephone",
        "func": lambda x: x["hv243a"]
    },
    "hv221": {
        "label": "Telephone (non-mobile)",
        "func": lambda x: x["hv221"]
    },
    "hv209": {
        "label": "Refrigerator",
        "func": lambda x: x["hv209"]
    },
    "sh110i": {
        "label": "Freezer",
        "func": lambda x: x["sh110i"]
    },
    "sh110j": {
        "label": "Electric generator/inverter",
        "func": lambda x: x["sh110j"]
    },
    "sh110k": {
        "label": "Washing machine",
        "func": lambda x: x["sh110k"]
    },
    "sh110l": {
        "label": "Computer/tablet computer",
        "func": lambda x: x["sh110l"]
    },
    "sh110m": {
        "label": "Photo camera",
        "func": lambda x: x["sh110m"]
    },
    "sh110n": {
        "label": "Video deck/DVD/VCD",
        "func": lambda x: x["sh110n"]
    },
    "sh110o": {
        "label": "Sewing machine",
        "func": lambda x: x["sh110o"]
    },
    "sh110p": {
        "label": "Bed",
        "func": lambda x: x["sh110p"]
    },
    "sh110q": {
        "label": "Table",
        "func": lambda x: x["sh110q"]
    },
    "sh110r": {
        "label": "Cabinet/cupboard",
        "func": lambda x: x["sh110r"]
    },
    "sh110s": {
        "label": "Access to internet in any device",
        "func": lambda x: x["sh110s"]
    },
    "hv243b": {
        "label": "Watch",
        "func": lambda x: x["hv243b"]
    },
    "hv210": {
        "label": "Bicycle",
        "func": lambda x: x["hv210"]
    },
    "hv211": {
        "label": "Motorcycle/scooter",
        "func": lambda x: x["hv211"]
    },
    "hv243c": {
        "label": "Animal-drawn cart",
        "func": lambda x: x["hv243c"]
    },
    "hv212": {
        "label": "Car/truck",
        "func": lambda x: x["hv212"]
    },
    "hv243d": {
        "label": "Boat with a motor",
        "func": lambda x: x["hv243d"]
    },
    # "hv243d_???": {
    #     "label": "Boat without a motor",
    #     "func": lambda x: "TODO"
    # },
    "hv247": {
        "label": "Bank account",
        "func": lambda x: x["hv247"]
    },
    "hv244": {
        "label": "Owns land suitable for agriculture",
        "func": lambda x: x["hv244"]
    },
    "hv201_11": {
        "label": "Source of drinking water: piped into dwelling",
        "func": lambda x: 1 if x["hv201"] == 11 else 0
    },
    "hv201_12": {
        "label": "Source of drinking water: piped into yard/plot",
        "func": lambda x: 1 if x["hv201"] == 12 else 0
    },
    "hv201_13": {
        "label": "Source of drinking water: public tap/standpipe",
        "func": lambda x: 1 if x["hv201"] == 13 else 0
    },
    "hv201_21": {
        "label": "Source of drinking water: tube well or borehole",
        "func": lambda x: 1 if x["hv201"] == 21 else 0
    },
    "hv201_31": {
        "label": "Source of drinking water: dug well protected",
        "func": lambda x: 1 if x["hv201"] == 31 else 0
    },
    "hv201_32": {
        "label": "Source of drinking water: dug well unprotected",
        "func": lambda x: 1 if x["hv201"] == 32 else 0
    },
    "hv201_41": {
        "label": "Source of drinking water: spring protected",
        "func": lambda x: 1 if x["hv201"] == 41 else 0
    },
    "hv201_42": {
        "label": "Source of drinking water: spring unprotected",
        "func": lambda x: 1 if x["hv201"] == 42 else 0
    },
    "hv201_51": {
        "label": "Source of drinking water: rainwater",
        "func": lambda x: 1 if x["hv201"] == 51 else 0
    },
    "hv201_61": {
        "label": "Source of drinking water: tanker truck",
        "func": lambda x: 1 if x["hv201"] == 61 else 0
    },
    "hv201_81": {
        "label": "Source of drinking water: river/dam/lake/ponds/stream/canal/channel",
        "func": lambda x: 1 if x["hv201"] == 81 or x["hv201"] == 43 else 0
    },
    "hv201_91": {
        "label": "Source of drinking water: bottled water",
        "func": lambda x: 1 if x["hv201"] == 91 or x["hv201"] == 71 else 0
    },
    "hv201_92": {
        "label": "Source of drinking water: sachet water",
        "func": lambda x: 1 if x["hv201"] == 92 or x["hv201"] == 72 else 0
    },
    "hv201_96": {
        "label": "Source of drinking water: other or cart with small tank",
        "func": lambda x: 1 if x["hv201"] == 96 or x["hv201"] == 62 else 0
    },
    "hv205_11": {
        "label": "Type of toilet facility: flush to piped sewer system",
        "func": lambda x: 1 if x["hv205"] == 11 else 0
    },
    "hv205_12": {
        "label": "Type of toilet facility: flush to septic tank",
        "func": lambda x: 1 if x["hv205"] == 12 else 0
    },
    "hv205_13": {
        "label": "Type of toilet facility: flush to pit latrine",
        "func": lambda x: 1 if x["hv205"] == 13 else 0
    },
    "hv205_14": {
        "label": "Type of toilet facility: flush to somewhere else",
        "func": lambda x: 1 if x["hv205"] == 14 else 0
    },
    "hv205_15": {
        "label": "Type of toilet facility: flush to don't know where",
        "func": lambda x: 1 if x["hv205"] == 15 else 0
    },
    "hv205_21": {
        "label": "Type of toilet facility: ventilated improved pit latrine",
        "func": lambda x: 1 if x["hv205"] == 21 else 0
    },
    "hv205_22": {
        "label": "Type of toilet facility: pit latrine with slab",
        "func": lambda x: 1 if x["hv205"] == 22 else 0
    },
    "hv205_23": {
        "label": "Type of toilet facility: pit latrine without slab/open pit",
        "func": lambda x: 1 if x["hv205"] == 23 else 0
    },
    "hv205_41": {
        "label": "Type of toilet facility: bucket toilet",
        "func": lambda x: 1 if x["hv205"] == 42 else 0
    },
    "hv205_51": {
        "label": "Type of toilet facility: hanging toilet/hanging latrine",
        "func": lambda x: 1 if x["hv205"] == 51 or x["hv205"] == 43 else 0
    },
    "hv205_61": {
        "label": "Type of toilet facility: no facility/bush/field",
        "func": lambda x: 1 if x["hv205"] == 61 or x["hv205"] == 31 else 0
    },
    "hv205_11_sh": {
        "label": "Type of toilet facility: flush to piped sewer system shared",
        "func": lambda x: 1 if x["hv205"] == 11 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_12_sh": {
        "label": "Type of toilet facility: flush to septic tank shared",
        "func": lambda x: 1 if x["hv205"] == 12 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_13_sh": {
        "label": "Type of toilet facility: flush to pit latrine shared",
        "func": lambda x: 1 if x["hv205"] == 13 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_14_sh": {
        "label": "Type of toilet facility: flush to somewhere else shared",
        "func": lambda x: 1 if x["hv205"] == 14 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_15_sh": {
        "label": "Type of toilet facility: flush to don't know where shared",
        "func": lambda x: 1 if x["hv205"] == 15 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_21_sh": {
        "label": "Type of toilet facility: ventilated improved pit latrine shared",
        "func": lambda x: 1 if x["hv205"] == 21 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_22_sh": {
        "label": "Type of toilet facility: pit latrine with slab shared",
        "func": lambda x: 1 if x["hv205"] == 22 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_23_sh": {
        "label": "Type of toilet facility: pit latrine without slab/open pit shared",
        "func": lambda x: 1 if x["hv205"] == 23 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_41_sh": {
        "label": "Type of toilet facility: bucket toilet shared",
        "func": lambda x: 1 if x["hv205"] == 42 and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv205_51_sh": {
        "label": "Type of toilet facility: hanging toilet/hanging latrine shared",
        "func": lambda x: 1 if (x["hv205"] == 51 or x["hv205"] == 43) and (x["hv225"] == 1 or x["hv225"] == 2) else 0
    },
    "hv226_1": {
        "label": "Type of cooking fuel: electricity",
        "func": lambda x: 1 if x["hv226"] == 1 else 0
    },
    "hv226_2": {
        "label": "Type of cooking fuel: LPG",
        "func": lambda x: 1 if x["hv226"] == 2 else 0
    },
    "hv226_3": {
        "label": "Type of cooking fuel: natural gas",
        "func": lambda x: 1 if x["hv226"] == 3 else 0
    },
    "hv226_5": {
        "label": "Type of cooking fuel: kerosene",
        "func": lambda x: 1 if x["hv226"] == 5 else 0
    },
    "hv226_6": {
        "label": "Type of cooking fuel: coal/lignite",
        "func": lambda x: 1 if x["hv226"] == 6 else 0
    },
    "hv226_7": {
        "label": "Type of cooking fuel: charcoal",
        "func": lambda x: 1 if x["hv226"] == 7 else 0
    },
    "hv226_8": {
        "label": "Type of cooking fuel: wood",
        "func": lambda x: 1 if x["hv226"] == 8 else 0
    },
    "hv226_9": {
        "label": "Type of cooking fuel: straw/shrubs/grass",
        "func": lambda x: 1 if x["hv226"] == 9 else 0
    },
    "hv226_10": {
        "label": "Type of cooking fuel: agricultural crop",
        "func": lambda x: 1 if x["hv226"] == 10 else 0
    },
    "hv226_95": {
        "label": "Type of cooking fuel: no food cooked in house",
        "func": lambda x: 1 if x["hv226"] == 95 else 0
    },
    "hv213_11": {
        "label": "Main floor material: earth/sand",
        "func": lambda x: 1 if x["hv213"] == 11 else 0
    },
    "hv213_12": {
        "label": "Main floor material: dung",
        "func": lambda x: 1 if x["hv213"] == 12 else 0
    },
    "hv213_21": {
        "label": "Main floor material: wood planks",
        "func": lambda x: 1 if x["hv213"] == 21 else 0
    },
    "hv213_31": {
        "label": "Main floor material: parquet/polished wood",
        "func": lambda x: 1 if x["hv213"] == 31 else 0
    },
    "hv213_33": {
        "label": "Main floor material: ceramic/marble/porcelain tiles/terrazo",
        "func": lambda x: 1 if x["hv213"] == 33 else 0
    },
    "hv213_34": {
        "label": "Main floor material: cement",
        "func": lambda x: 1 if x["hv213"] == 34 else 0
    },
    "hv213_35": {
        "label": "Main floor material: woolen carpets/synthetic carpet",
        "func": lambda x: 1 if x["hv213"] == 35 else 0
    },
    "hv213_36": {
        "label": "Main floor material: linoleum/rubber carpet",
        "func": lambda x: 1 if x["hv213"] == 36 else 0
    },
    "hv213_96": {
        "label": "Main floor material: other",
        "func": lambda x: 1 if x["hv213"] == 96 else 0
    },
    "hv215_11": {
        "label": "Main roof material: no roof",
        "func": lambda x: 1 if x["hv215"] == 11 else 0
    },
    "hv215_12": {
        "label": "Main roof material: thatch/palm leaf",
        "func": lambda x: 1 if x["hv215"] == 12 else 0
    },
    "hv215_21": {
        "label": "Main roof material: rustic mat",
        "func": lambda x: 1 if x["hv215"] == 21 else 0
    },
    "hv215_22": {
        "label": "Main roof material: palm/bamboo",
        "func": lambda x: 1 if x["hv215"] == 22 else 0
    },
    "hv215_23": {
        "label": "Main roof material: wood planks",
        "func": lambda x: 1 if x["hv215"] == 23 else 0
    },
    "hv215_31": {
        "label": "Main roof material: metal",
        "func": lambda x: 1 if x["hv215"] == 31 else 0
    },
    "hv215_32": {
        "label": "Main roof material: wood",
        "func": lambda x: 1 if x["hv215"] == 32 else 0
    },
    "hv215_33": {
        "label": "Main roof material: calamine/cement fiber",
        "func": lambda x: 1 if x["hv215"] == 33 else 0
    },
    "hv215_34": {
        "label": "Main roof material: ceramic tiles",
        "func": lambda x: 1 if x["hv215"] == 34 else 0
    },
    "hv215_35": {
        "label": "Main roof material: cement",
        "func": lambda x: 1 if x["hv215"] == 35 else 0
    },
    "hv215_36": {
        "label": "Main roof material: roofing shingles",
        "func": lambda x: 1 if x["hv215"] == 36 else 0
    },
    "hv215_37": {
        "label": "Main roof material: asbestos/slate roofing sheets",
        "func": lambda x: 1 if x["hv215"] == 37 else 0
    },
    "hv215_96": {
        "label": "Main roof material: other",
        "func": lambda x: 1 if x["hv215"] == 96 else 0
    },
    "hv214_11": {
        "label": "Main wall material: no walls",
        "func": lambda x: 1 if x["hv214"] == 11 else 0
    },
    "hv214_12": {
        "label": "Main wall material: cane/palm/trunks",
        "func": lambda x: 1 if x["hv214"] == 12 else 0
    },
    "hv214_13": {
        "label": "Main wall material: dirt",
        "func": lambda x: 1 if x["hv214"] == 13 else 0
    },
    "hv214_21": {
        "label": "Main wall material: bamboo with mud",
        "func": lambda x: 1 if x["hv214"] == 21 else 0
    },
    "hv214_22": {
        "label": "Main wall material: stone with mud",
        "func": lambda x: 1 if x["hv214"] == 22 else 0
    },
    "hv214_23": {
        "label": "Main wall material: uncovered adobe",
        "func": lambda x: 1 if x["hv214"] == 23 else 0
    },
    "hv214_24": {
        "label": "Main wall material: plywood",
        "func": lambda x: 1 if x["hv214"] == 24 else 0
    },
    "hv214_26": {
        "label": "Main wall material: reused wood",
        "func": lambda x: 1 if x["hv214"] == 26 else 0
    },
    "hv214_31": {
        "label": "Main wall material: cement",
        "func": lambda x: 1 if x["hv214"] == 31 else 0
    },
    "hv214_32": {
        "label": "Main wall material: stone with lime/cement",
        "func": lambda x: 1 if x["hv214"] == 32 else 0
    },
    "hv214_33": {
        "label": "Main wall material: bricks",
        "func": lambda x: 1 if x["hv214"] == 33 else 0
    },
    "hv214_34": {
        "label": "Main wall material: cement blocks",
        "func": lambda x: 1 if x["hv214"] == 34 else 0
    },
    "hv214_35": {
        "label": "Main wall material: covered adobe",
        "func": lambda x: 1 if x["hv214"] == 35 else 0
    },
    "hv214_36": {
        "label": "Main wall material: wood planks/shingles",
        "func": lambda x: 1 if x["hv214"] == 36 else 0
    },
    "hv214_96": {
        "label": "Main wall material: other",
        "func": lambda x: 1 if x["hv214"] == 96 else 0
    },
}
