"""
Atom-name / residue-property dictionaries for the Pocket-3D feature extractor.

Pure data module — no I/O, no NumPy, no dependencies. Imported by
`build_pocket3d_features.py`. Keep this small and readable; any change here
alters the feature semantics and invalidates the cached npz.

References:
  - PDB CRO (matured GFP chromophore) canonical atom names from RCSB
    Chemical Component Dictionary. Tripeptide variants (SYG/TYG/GYG/...) share
    the phenol + imidazolinone core atoms but differ in the first residue's
    side chain (SYG=Ser OG1, TYG=Thr OG1+CG1, CYG=Cys SG, etc.).
"""
from __future__ import annotations

# ─── Chromophore HETATM residue names ────────────────────────────────────────
# All chromophore HETATM 3-letter codes encountered in minimized/grafted PDBs.
# Empirically derived from a HETATM scan over `data/structure/minimized/*.pdb`
# (see build_pocket3d_features.ipynb diagnostic). Includes:
#   - matured GFP-like generic (CRO/CR2/CR8/CR7/CRU/CRQ/CRF)
#   - explicit tripeptide codes (TYG/SYG/GYG/MYG/CYG/QYG/HYG/etc.)
#   - red-FP variants (NRQ/NRP — mCherry/mScarlet/DsRed family)
#   - blue/cyan/yellow variants (SWG/SHG — TagBFP/Sirius family; CH6/CH7)
#   - misc 3- and 4-letter PDB codes (4M9/5SQ/0WZ/7R0 — newer depositions)
# `has_pocket` self-polices: any HETATM here that lacks phenol-OH + ≥3 phenol
# ring atoms simply gets has_pocket=0 in extract_for_slug().
CHROM_HETATM_NAMES = {
    # Matured GFP-like generic codes
    "CRO", "CR2", "CR8", "CR7", "CRU", "CRQ", "CRF",
    # Explicit tripeptide codes (X-Y-G or X-W-G)
    "GYG", "TYG", "SYG", "MYG", "CYG", "HYG", "AYG", "EYG",
    "QYG", "DYG", "NYG", "FYG", "KYG", "LYG", "RYG",
    "TWG", "SWG", "AWG", "QWG", "MWG",
    "SHG", "THG", "AHG", "LHG", "QHG",
    "GYS", "GYC",
    # Red FP chromophore variants
    "NRQ", "NRP",
    # Blue/cyan/yellow variants
    "CH6", "CH7", "CFY", "WCR", "OFM", "CIV",
    # Newer / less common PDB chromophore codes
    "4M9", "5SQ", "0WZ", "7R0", "BJO", "BJF", "PIA", "CCY", "IIC",
    "GZG",
}

# Phenol-ring atom names (from Tyr66 side chain, shared across all tripeptides
# because all GFP-family chromophores retain the tyrosine aromatic ring).
PHENOL_RING_ATOMS = ("CG2", "CD1", "CD2", "CE1", "CE2", "CZ")
PHENOL_OH_ATOM    = "OH"

# Bridge atoms linking phenol to imidazolinone (used for τ dihedral).
BRIDGE_ATOMS = ("CA2", "CB2")

# Imidazolinone ring atoms (5-membered cyclic imidazolinone formed in maturation).
# Name conventions vary slightly across RCSB entries; we accept several aliases.
IMID_RING_ATOMS = ("CA2", "C1", "N2", "C2", "N3")
IMID_RING_ATOMS_ALT = ("CA2", "C1", "N2", "CA3", "N3")  # some PDBs use CA3 for C2

# Full expected heavy-atom set per tripeptide (for `atom_completeness` sanity).
# Minimum = phenol + bridge + imidazolinone core = ~13 atoms.
_PHENOL_BRIDGE_IMID = set(PHENOL_RING_ATOMS) | {PHENOL_OH_ATOM} | set(BRIDGE_ATOMS) | set(IMID_RING_ATOMS)

_CHROM_BACKBONE_MIN = _PHENOL_BRIDGE_IMID | {"N1", "CA1", "CA3", "C3"}

CHROM_EXPECTED_ATOMS: dict[str, set[str]] = {
    # Matured generic codes — minimal backbone + ring core
    "CRO": _CHROM_BACKBONE_MIN | {"OG1"},                                       # Ser65-Tyr66-Gly67 matured
    "CR2": _CHROM_BACKBONE_MIN,                                                 # DsRed-family variant
    "CR8": _CHROM_BACKBONE_MIN,                                                 # red-shifted variant
    "CR7": _CHROM_BACKBONE_MIN,
    "CRU": _CHROM_BACKBONE_MIN,
    "CRQ": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "CD1N", "OE1", "NE2"},          # Gln-derived
    "CRF": _CHROM_BACKBONE_MIN,
    # Explicit X-Y-G tripeptide codes
    "GYG": _CHROM_BACKBONE_MIN,
    "TYG": _CHROM_BACKBONE_MIN | {"OG1", "CG1"},                                # Thr-Tyr-Gly (EGFP)
    "SYG": _CHROM_BACKBONE_MIN | {"OG1"},                                       # Ser-Tyr-Gly (avGFP)
    "CYG": _CHROM_BACKBONE_MIN | {"SG"},                                        # Cys-Tyr-Gly
    "HYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "ND1", "NE2"},                  # His-Tyr-Gly
    "AYG": _CHROM_BACKBONE_MIN | {"CB1"},                                       # Ala-Tyr-Gly
    "EYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "OE1", "OE2"},
    "MYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "SD", "CE"},
    "QYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "CD1N", "OE1", "NE2"},
    "DYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "OD1", "OD2"},
    "NYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "OD1", "ND2"},
    "FYG": _CHROM_BACKBONE_MIN | {"CB1"},
    "KYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "CD1N", "CE", "NZ"},
    "LYG": _CHROM_BACKBONE_MIN | {"CB1"},
    "RYG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "NE", "NH1", "NH2"},
    # X-W-G tripeptides (Trp at position 2 — different ring system!)
    "TWG": _CHROM_BACKBONE_MIN | {"OG1", "CG1"},
    "SWG": _CHROM_BACKBONE_MIN | {"OG1"},                                       # Ser-Trp-Gly (TagBFP)
    "AWG": _CHROM_BACKBONE_MIN | {"CB1"},
    "QWG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "CD1N", "OE1", "NE2"},
    "MWG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "SD", "CE"},
    # X-H-G tripeptides
    "SHG": _CHROM_BACKBONE_MIN | {"OG1"},                                       # Ser-His-Gly (Sirius)
    "THG": _CHROM_BACKBONE_MIN | {"OG1", "CG1"},
    "AHG": _CHROM_BACKBONE_MIN | {"CB1"},
    "LHG": _CHROM_BACKBONE_MIN | {"CB1"},
    "QHG": _CHROM_BACKBONE_MIN | {"CB1", "CG1", "CD1N", "OE1", "NE2"},
    "GYS": _CHROM_BACKBONE_MIN | {"OG1"},                                       # Gly-Tyr-Ser variant
    "GYC": _CHROM_BACKBONE_MIN | {"SG"},
    # Red FP chromophore variants — extended π-system, minimum check is core only
    "NRQ": _CHROM_BACKBONE_MIN,                                                 # mCherry/mScarlet/DsRed mature
    "NRP": _CHROM_BACKBONE_MIN,                                                 # NRQ precursor
    # Blue/cyan/yellow / misc variants — use core only
    "CH6": _CHROM_BACKBONE_MIN, "CH7": _CHROM_BACKBONE_MIN,
    "CFY": _CHROM_BACKBONE_MIN, "WCR": _CHROM_BACKBONE_MIN,
    "OFM": _CHROM_BACKBONE_MIN, "CIV": _CHROM_BACKBONE_MIN,
    "4M9": _CHROM_BACKBONE_MIN, "5SQ": _CHROM_BACKBONE_MIN,
    "0WZ": _CHROM_BACKBONE_MIN, "7R0": _CHROM_BACKBONE_MIN,
    "BJO": _CHROM_BACKBONE_MIN, "BJF": _CHROM_BACKBONE_MIN,
    "PIA": _CHROM_BACKBONE_MIN, "CCY": _CHROM_BACKBONE_MIN,
    "IIC": _CHROM_BACKBONE_MIN, "GZG": _CHROM_BACKBONE_MIN,
}

# ─── Side-chain polar heavy atoms per residue ────────────────────────────────
# Used for the "named-atom nearest distance" features in Block B.
# Includes only heavy atoms that can donate/accept H-bonds or carry formal charge.
SIDECHAIN_POLAR_ATOMS: dict[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),                      # carboxylate
    "GLU": ("OE1", "OE2"),                      # carboxylate
    "LYS": ("NZ",),                             # ammonium
    "ARG": ("NE", "NH1", "NH2"),                # guanidinium
    "HIS": ("ND1", "NE2"),                      # imidazole
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "CYS": ("SG",),
    "ASN": ("OD1", "ND2"),
    "GLN": ("OE1", "NE2"),
    "TRP": ("NE1",),
}

# Formal charge approximation for electrostatic proxy (pH ≈ 7.0).
FORMAL_CHARGE: dict[str, float] = {
    "ARG": +1.0,
    "LYS": +1.0,
    "HIS": +0.5,   # partially protonated at physiological pH
    "ASP": -1.0,
    "GLU": -1.0,
    # all others default to 0.0
}

# ─── Three-letter to category map ────────────────────────────────────────────
HYDROPHOBIC = {"ALA", "ILE", "LEU", "MET", "PHE", "PRO", "VAL", "TRP"}
POLAR       = {"SER", "THR", "ASN", "GLN", "CYS"}
POSITIVE    = {"LYS", "ARG", "HIS"}
NEGATIVE    = {"ASP", "GLU"}
AROMATIC    = {"PHE", "TYR", "TRP", "HIS"}           # note: His overlaps POSITIVE
HBOND_DONOR = {"ASN", "GLN", "SER", "THR", "TYR", "LYS", "ARG", "HIS", "TRP", "CYS"}
HBOND_ACCEPT= {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "HIS", "CYS"}
SPECIAL_GLY = {"GLY"}

# Chromophore residue names also map to their parent "Y"-like category for
# shell composition (Tyr-like aromatic) — we exclude them from shells anyway.
_CHROM_TO_CAT = {name: AROMATIC for name in CHROM_HETATM_NAMES}


def aa_is(res3: str, category: str) -> int:
    """Return 1 if the given 3-letter residue belongs to the named category.

    Accepted categories: hydrophobic / polar / positive / negative / aromatic /
    hbond_donor / hbond_acceptor / special (Gly).
    """
    res3 = res3.upper()
    if res3 in CHROM_HETATM_NAMES:
        res3 = "TYR"  # chromophore HETATM treated as parent Tyr
    table = {
        "hydrophobic":    HYDROPHOBIC,
        "polar":          POLAR,
        "positive":       POSITIVE,
        "negative":       NEGATIVE,
        "aromatic":       AROMATIC,
        "hbond_donor":    HBOND_DONOR,
        "hbond_acceptor": HBOND_ACCEPT,
        "special":        SPECIAL_GLY,
    }
    return 1 if res3 in table[category] else 0


# Exposed category list (fixed order for deterministic feature-name output).
CATEGORIES = (
    "hydrophobic", "polar", "positive", "negative",
    "aromatic", "hbond_donor", "hbond_acceptor", "special",
)
