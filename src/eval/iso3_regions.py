"""ISO3 country code → UN M49 subregion mapping.

used for bloc-discovery eval: cluster graph-JEPA's country embeddings and
score against this ground-truth partition. UN M49 subregions are the
standard geographic classification (17 subregions, every country gets
exactly one assignment), close to but not identical to trade blocs.

source: UN Statistics Division M49 standard, manually compiled for the 227
BACI ISO3 codes. covers all of BACI's countries; non-trade entities
(e.g., Antarctica AIA codes) get assigned to their geographic region
even if they're not bloc participants.
"""
from __future__ import annotations

# 17 UN M49 subregions
SUBREGIONS = [
    "Northern Africa",
    "Sub-Saharan Africa",
    "Latin America and the Caribbean",
    "Northern America",
    "Central Asia",
    "Eastern Asia",
    "South-eastern Asia",
    "Southern Asia",
    "Western Asia",
    "Eastern Europe",
    "Northern Europe",
    "Southern Europe",
    "Western Europe",
    "Australia and New Zealand",
    "Melanesia",
    "Micronesia",
    "Polynesia",
]

ISO3_TO_SUBREGION: dict[str, str] = {
    # Northern Africa
    "DZA": "Northern Africa", "EGY": "Northern Africa", "LBY": "Northern Africa",
    "MAR": "Northern Africa", "SDN": "Northern Africa", "TUN": "Northern Africa",
    "ESH": "Northern Africa",
    # Sub-Saharan Africa
    "AGO": "Sub-Saharan Africa", "BDI": "Sub-Saharan Africa", "BEN": "Sub-Saharan Africa",
    "BFA": "Sub-Saharan Africa", "BWA": "Sub-Saharan Africa", "CAF": "Sub-Saharan Africa",
    "CIV": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa", "COD": "Sub-Saharan Africa",
    "COG": "Sub-Saharan Africa", "COM": "Sub-Saharan Africa", "CPV": "Sub-Saharan Africa",
    "DJI": "Sub-Saharan Africa", "ERI": "Sub-Saharan Africa", "ETH": "Sub-Saharan Africa",
    "GAB": "Sub-Saharan Africa", "GHA": "Sub-Saharan Africa", "GIN": "Sub-Saharan Africa",
    "GMB": "Sub-Saharan Africa", "GNB": "Sub-Saharan Africa", "GNQ": "Sub-Saharan Africa",
    "KEN": "Sub-Saharan Africa", "LBR": "Sub-Saharan Africa", "LSO": "Sub-Saharan Africa",
    "MDG": "Sub-Saharan Africa", "MLI": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa",
    "MRT": "Sub-Saharan Africa", "MUS": "Sub-Saharan Africa", "MWI": "Sub-Saharan Africa",
    "MYT": "Sub-Saharan Africa", "NAM": "Sub-Saharan Africa", "NER": "Sub-Saharan Africa",
    "NGA": "Sub-Saharan Africa", "REU": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "SEN": "Sub-Saharan Africa", "SHN": "Sub-Saharan Africa", "SLE": "Sub-Saharan Africa",
    "SOM": "Sub-Saharan Africa", "SSD": "Sub-Saharan Africa", "STP": "Sub-Saharan Africa",
    "SWZ": "Sub-Saharan Africa", "SYC": "Sub-Saharan Africa", "TCD": "Sub-Saharan Africa",
    "TGO": "Sub-Saharan Africa", "TZA": "Sub-Saharan Africa", "UGA": "Sub-Saharan Africa",
    "ZAF": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa", "ZWE": "Sub-Saharan Africa",
    # Latin America and the Caribbean
    "ABW": "Latin America and the Caribbean", "AIA": "Latin America and the Caribbean",
    "ANT": "Latin America and the Caribbean", "ARG": "Latin America and the Caribbean",
    "ATG": "Latin America and the Caribbean", "BES": "Latin America and the Caribbean",
    "BHS": "Latin America and the Caribbean", "BLM": "Latin America and the Caribbean",
    "BLZ": "Latin America and the Caribbean", "BOL": "Latin America and the Caribbean",
    "BRA": "Latin America and the Caribbean", "BRB": "Latin America and the Caribbean",
    "CHL": "Latin America and the Caribbean", "COL": "Latin America and the Caribbean",
    "CRI": "Latin America and the Caribbean", "CUB": "Latin America and the Caribbean",
    "CUW": "Latin America and the Caribbean", "CYM": "Latin America and the Caribbean",
    "DMA": "Latin America and the Caribbean", "DOM": "Latin America and the Caribbean",
    "ECU": "Latin America and the Caribbean", "FLK": "Latin America and the Caribbean",
    "GLP": "Latin America and the Caribbean", "GRD": "Latin America and the Caribbean",
    "GTM": "Latin America and the Caribbean", "GUF": "Latin America and the Caribbean",
    "GUY": "Latin America and the Caribbean", "HND": "Latin America and the Caribbean",
    "HTI": "Latin America and the Caribbean", "JAM": "Latin America and the Caribbean",
    "KNA": "Latin America and the Caribbean", "LCA": "Latin America and the Caribbean",
    "MAF": "Latin America and the Caribbean", "MEX": "Latin America and the Caribbean",
    "MSR": "Latin America and the Caribbean", "MTQ": "Latin America and the Caribbean",
    "NIC": "Latin America and the Caribbean", "PAN": "Latin America and the Caribbean",
    "PER": "Latin America and the Caribbean", "PRI": "Latin America and the Caribbean",
    "PRY": "Latin America and the Caribbean", "SLV": "Latin America and the Caribbean",
    "SUR": "Latin America and the Caribbean", "SXM": "Latin America and the Caribbean",
    "TCA": "Latin America and the Caribbean", "TTO": "Latin America and the Caribbean",
    "URY": "Latin America and the Caribbean", "VCT": "Latin America and the Caribbean",
    "VEN": "Latin America and the Caribbean", "VGB": "Latin America and the Caribbean",
    "VIR": "Latin America and the Caribbean",
    # Northern America
    "BMU": "Northern America", "CAN": "Northern America", "GRL": "Northern America",
    "SPM": "Northern America", "USA": "Northern America",
    # Central Asia
    "KAZ": "Central Asia", "KGZ": "Central Asia", "TJK": "Central Asia",
    "TKM": "Central Asia", "UZB": "Central Asia",
    # Eastern Asia
    "CHN": "Eastern Asia", "HKG": "Eastern Asia", "JPN": "Eastern Asia",
    "KOR": "Eastern Asia", "MAC": "Eastern Asia", "MNG": "Eastern Asia",
    "PRK": "Eastern Asia", "TWN": "Eastern Asia",
    # South-eastern Asia
    "BRN": "South-eastern Asia", "IDN": "South-eastern Asia", "KHM": "South-eastern Asia",
    "LAO": "South-eastern Asia", "MMR": "South-eastern Asia", "MYS": "South-eastern Asia",
    "PHL": "South-eastern Asia", "SGP": "South-eastern Asia", "THA": "South-eastern Asia",
    "TLS": "South-eastern Asia", "VNM": "South-eastern Asia",
    # Southern Asia
    "AFG": "Southern Asia", "BGD": "Southern Asia", "BTN": "Southern Asia",
    "IND": "Southern Asia", "IRN": "Southern Asia", "LKA": "Southern Asia",
    "MDV": "Southern Asia", "NPL": "Southern Asia", "PAK": "Southern Asia",
    # Western Asia
    "ARE": "Western Asia", "ARM": "Western Asia", "AZE": "Western Asia",
    "BHR": "Western Asia", "CYP": "Western Asia", "GEO": "Western Asia",
    "IRQ": "Western Asia", "ISR": "Western Asia", "JOR": "Western Asia",
    "KWT": "Western Asia", "LBN": "Western Asia", "OMN": "Western Asia",
    "PSE": "Western Asia", "QAT": "Western Asia", "SAU": "Western Asia",
    "SYR": "Western Asia", "TUR": "Western Asia", "YEM": "Western Asia",
    # Eastern Europe
    "BGR": "Eastern Europe", "BLR": "Eastern Europe", "CZE": "Eastern Europe",
    "HUN": "Eastern Europe", "MDA": "Eastern Europe", "POL": "Eastern Europe",
    "ROU": "Eastern Europe", "RUS": "Eastern Europe", "SVK": "Eastern Europe",
    "UKR": "Eastern Europe",
    # Northern Europe
    "ALA": "Northern Europe", "DNK": "Northern Europe", "EST": "Northern Europe",
    "FIN": "Northern Europe", "FRO": "Northern Europe", "GBR": "Northern Europe",
    "GGY": "Northern Europe", "IMN": "Northern Europe", "IRL": "Northern Europe",
    "ISL": "Northern Europe", "JEY": "Northern Europe", "LTU": "Northern Europe",
    "LVA": "Northern Europe", "NOR": "Northern Europe", "SJM": "Northern Europe",
    "SWE": "Northern Europe",
    # Southern Europe
    "ALB": "Southern Europe", "AND": "Southern Europe", "BIH": "Southern Europe",
    "ESP": "Southern Europe", "GIB": "Southern Europe", "GRC": "Southern Europe",
    "HRV": "Southern Europe", "ITA": "Southern Europe", "MKD": "Southern Europe",
    "MLT": "Southern Europe", "MNE": "Southern Europe", "PRT": "Southern Europe",
    "SMR": "Southern Europe", "SRB": "Southern Europe", "SVN": "Southern Europe",
    "VAT": "Southern Europe",
    # Western Europe
    "AUT": "Western Europe", "BEL": "Western Europe", "CHE": "Western Europe",
    "DEU": "Western Europe", "FRA": "Western Europe", "LIE": "Western Europe",
    "LUX": "Western Europe", "MCO": "Western Europe", "NLD": "Western Europe",
    # Australia and New Zealand
    "AUS": "Australia and New Zealand", "NZL": "Australia and New Zealand",
    "NFK": "Australia and New Zealand",
    # Melanesia
    "FJI": "Melanesia", "NCL": "Melanesia", "PNG": "Melanesia",
    "SLB": "Melanesia", "VUT": "Melanesia",
    # Micronesia
    "FSM": "Micronesia", "GUM": "Micronesia", "KIR": "Micronesia",
    "MHL": "Micronesia", "MNP": "Micronesia", "NRU": "Micronesia",
    "PLW": "Micronesia",
    # Polynesia
    "ASM": "Polynesia", "COK": "Polynesia", "NIU": "Polynesia",
    "PCN": "Polynesia", "PYF": "Polynesia", "TKL": "Polynesia",
    "TON": "Polynesia", "TUV": "Polynesia", "WLF": "Polynesia",
    "WSM": "Polynesia",
    # additional codes that appear in BACI
    "CCK": "Australia and New Zealand",  # Cocos (Keeling) Islands - Australian territory
    "CXR": "Australia and New Zealand",  # Christmas Island - Australian territory
    "IOT": "Sub-Saharan Africa",         # British Indian Ocean Territory
    "SCG": "Southern Europe",            # legacy: Serbia and Montenegro pre-2006
}


def label_iso3_list(iso3_codes: list[str]) -> tuple[list[str], list[str]]:
    """return (labels, missing) where labels[i] is the subregion for iso3_codes[i]
    (or None if not in our table), and missing is a list of unmapped codes."""
    labels: list[str] = []
    missing: list[str] = []
    for code in iso3_codes:
        sub = ISO3_TO_SUBREGION.get(code)
        if sub is None:
            missing.append(code)
            labels.append("Unknown")
        else:
            labels.append(sub)
    return labels, missing


# coarser partition: continent (5 buckets). easier ground truth for a bloc
# discovery test — "did the model recover the basic geographic structure of trade?"
SUBREGION_TO_CONTINENT: dict[str, str] = {
    "Northern Africa": "Africa",
    "Sub-Saharan Africa": "Africa",
    "Latin America and the Caribbean": "Americas",
    "Northern America": "Americas",
    "Central Asia": "Asia",
    "Eastern Asia": "Asia",
    "South-eastern Asia": "Asia",
    "Southern Asia": "Asia",
    "Western Asia": "Asia",
    "Eastern Europe": "Europe",
    "Northern Europe": "Europe",
    "Southern Europe": "Europe",
    "Western Europe": "Europe",
    "Australia and New Zealand": "Oceania",
    "Melanesia": "Oceania",
    "Micronesia": "Oceania",
    "Polynesia": "Oceania",
}

CONTINENTS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]


def label_iso3_list_continent(iso3_codes: list[str]) -> tuple[list[str], list[str]]:
    """coarser partition into 5 continents."""
    sub_labels, missing = label_iso3_list(iso3_codes)
    cont_labels = [SUBREGION_TO_CONTINENT.get(s, "Unknown") for s in sub_labels]
    return cont_labels, missing
