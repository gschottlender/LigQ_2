from __future__ import annotations

SEEDS = (42, 10, 27, 3, 8)
RAW_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0, 90.0, 80.0, 50.0)
S1_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0, 90.0)
RAW_NEIGHBOR_COUNTS = (1, 2, 3, 4, 5, 10, 15, 20)
S1_NEIGHBOR_COUNTS = (1, 2, 3, 4, 5, 10, 15)
PLOTTED_NEIGHBOR_COUNTS = (2, 3, 5, 10, 15)

METHODS = {
    "morgan_1024_r2": "tanimoto",
    "ap_rdkit": "tanimoto",
    "chemberta_zinc_base_768": "cosine",
    "maccs": "tanimoto",
    "rdkit_1024": "tanimoto",
    "topological_torsion_rdkit_1024": "tanimoto",
    "morgan_feature_1024_r2": "tanimoto",
}

PRETTY_METHODS = {
    "morgan_1024_r2": "ECFP4 (1024 bits)",
    "morgan_feature_1024_r2": "FCFP4 (1024 bits)",
    "ap_rdkit": "Atom Pair",
    "chemberta_zinc_base_768": "ChemBERTa",
    "maccs": "MACCS",
    "rdkit_1024": "RDKit Path",
    "topological_torsion_rdkit_1024": "Topological Torsion",
}

FAMILIES = {
    "Quinasas": {
        "Tirosina (receptor)": ["csf1r", "egfr", "fgfr1", "igf1r", "kit", "met", "vgfr2"],
        "Tirosina (no receptor)": ["abl1", "fak1", "jak2", "lck", "src"],
        "Ser/Thr (incl. MAPK)": ["akt1", "akt2", "braf", "cdk2", "plk1", "rock1", "wee1", "tgfr1", "kpcb", "mk01", "mk10", "mk14", "mp2k1", "mapk2"],
    },
    "Proteasas": {
        "Serina": ["fa10", "fa7", "thrb", "try1", "tryb1", "urok", "dpp4"],
        "Aspártico": ["bace1", "hivpr", "reni"],
        "Cisteína": ["casp3"],
        "Metalo": ["ace", "ada17", "mmp13", "lkha4"],
    },
    "Receptores nucleares": {
        "Esteroideos": ["andr", "gcr", "mcr", "prgr", "esr1", "esr2"],
        "PPAR": ["ppara", "ppard", "pparg"],
        "Otros": ["rxra", "thb"],
    },
    "GPCR": {
        "Adenosina": ["aa2ar"],
        "Adrenérgicos": ["adrb1", "adrb2"],
        "Dopaminérgico": ["drd3"],
        "Quimiocina": ["cxcr4"],
    },
    "Canales iónicos": {"Glutamato (ionotrópicos)": ["gria2", "grik1"]},
    "Citocromo P450": {"Familia CYP": ["cp2c9", "cp3a4"]},
    "Otras enzimas": {
        "Oxidoreductasas": ["aofb", "aldr", "dhi1", "dyr", "hmdh", "inha", "nos1", "pgh1", "pgh2", "pyrd"],
        "Hidrolasas": ["aces", "ampc", "def", "glcm", "hdac2", "hdac8", "nram", "pa2ga", "pde5a", "ptn1", "sahh"],
        "Liasas (anhidrasas carbónicas)": ["cah2"],
        "Transferasas": ["comt", "fnta", "fpps", "hxk4", "kith", "parp1", "pnph", "pur2", "tysy", "hivrt", "hivint"],
        "Otros (isomerasa/etc)": ["fkb1a"],
    },
    "Misceláneos": {
        "Chaperona (Hsp90)": ["hs90a"],
        "Prot. unión ác. graso": ["fabp4"],
        "Integrina": ["ital"],
        "Motor proteico": ["kif11"],
        "Regulador apoptosis": ["xiap"],
    },
}

FAMILY_ORDER = tuple(FAMILIES)
FAMILY_LABELS = {
    "Quinasas": "Kinases", "Proteasas": "Proteases",
    "Receptores nucleares": "Nuclear receptors", "GPCR": "GPCRs",
    "Canales iónicos": "Ion channels", "Citocromo P450": "Cytochrome P450",
    "Otras enzimas": "Other enzymes", "Misceláneos": "Miscellaneous",
}

BSI_TARGETS = ("akt1", "akt2", "cdk2", "kpcb", "mapk2", "mk01", "mk10", "mk14", "mp2k1", "plk1", "rock1", "tgfr1", "wee1")
BSI_S1_TARGETS = ("akt1", "cdk2", "mk01", "mk10", "mk14", "mp2k1", "plk1", "rock1", "tgfr1", "wee1")
