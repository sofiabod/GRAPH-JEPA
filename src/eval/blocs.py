"""ground-truth economic-bloc labels for unsupervised cluster recovery.

iso3 codes mapped to bloc memberships. countries with no bloc affiliation are
labeled "NONE" — they form a residual class against which cluster purity is
measured. countries in multiple blocs are resolved via BLOC_PRIORITY (single
hard label per country).

source: standard regional trade agreements as of 2020.
"""

# bloc → list of iso3 codes
BLOC_LABELS: dict[str, list[str]] = {
    "EU": [
        "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
    ],
    "USMCA": ["USA", "CAN", "MEX"],
    "ASEAN": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
    "GCC": ["SAU", "ARE", "KWT", "OMN", "QAT", "BHR"],
    "EAEU": ["RUS", "BLR", "KAZ", "ARM", "KGZ"],
    "MERCOSUR": ["ARG", "BRA", "PRY", "URY", "VEN", "BOL"],
    "ECOWAS": [
        "BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI",
        "NER", "NGA", "SEN", "SLE", "TGO",
    ],
    "SADC": [
        "AGO", "BWA", "COD", "SWZ", "LSO", "MDG", "MWI", "MUS", "MOZ", "NAM",
        "ZAF", "SYC", "TZA", "ZMB", "ZWE",
    ],
    "SAARC": ["AFG", "BGD", "BTN", "IND", "MDV", "NPL", "PAK", "LKA"],
    "EFTA_OTHER": ["CHE", "NOR", "ISL"],
}

# resolution order when a country appears in multiple blocs (first match wins)
BLOC_PRIORITY: list[str] = [
    "EU", "USMCA", "ASEAN", "GCC", "EAEU", "MERCOSUR",
    "ECOWAS", "SADC", "SAARC", "EFTA_OTHER",
]


def country_bloc(iso3: str) -> str:
    """return bloc name for an iso3 code, or 'NONE' if unaligned."""
    for bloc in BLOC_PRIORITY:
        if iso3 in BLOC_LABELS[bloc]:
            return bloc
    return "NONE"


def bloc_distribution(iso_list: list[str]) -> dict[str, int]:
    """count countries per bloc in the given iso list."""
    counts: dict[str, int] = {}
    for iso in iso_list:
        b = country_bloc(iso)
        counts[b] = counts.get(b, 0) + 1
    return counts


def labels_for_isos(iso_list: list[str]) -> list[str]:
    """parallel array of bloc labels for the given iso list."""
    return [country_bloc(iso) for iso in iso_list]
