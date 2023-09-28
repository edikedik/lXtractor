"""
A module encompassing various settings of lXtractor objects.
"""
from __future__ import annotations

import typing as t
from collections import namedtuple
from dataclasses import dataclass
from enum import IntFlag

Bounds = t.NamedTuple("Bounds", [("lower", float), ("upper", float)])
Separators = namedtuple(
    "Separators", ["list", "chain", "dom", "uni_pdb", "str", "start_end"]
)
Sep = Separators(",", ":", "::", "_", "--", "|")
EMPTY_STRUCTURE_ID: str = "XXXX"
EMPTY_CHAIN_ID: str = "X"
UNK_NAME: str = "Unk"
STRUCTURE_EXT = (".cif", ".pdb", ".mmtf")
STRUCTURE_FMT = tuple(x.removeprefix(".") for x in STRUCTURE_EXT)

_AminoAcids = [
    ("ALA", "A"),
    ("CYS", "C"),
    ("THR", "T"),
    ("GLU", "E"),
    ("ASP", "D"),
    ("PHE", "F"),
    ("TRP", "W"),
    ("ILE", "I"),
    ("VAL", "V"),
    ("LEU", "L"),
    ("LYS", "K"),
    ("MET", "M"),
    ("ASN", "N"),
    ("GLN", "Q"),
    ("SER", "S"),
    ("ARG", "R"),
    ("TYR", "Y"),
    ("HIS", "H"),
    ("PRO", "P"),
    ("GLY", "G"),
]
SOLVENTS = (
    "1PE",
    "2HT",
    "2PE",
    "7PE",
    "ACT",
    "ACT",
    "BME",
    "BTB",
    "BU3",
    "BUD",
    "CIT",
    "COM",
    "CXS",
    "DIO",
    "DMS",
    "DOD",
    "DTD",
    "DTT",
    "DTV",
    "DVT",
    "EDO",
    "EOH",
    "EPE",
    "FLC",
    "FMT",
    "GBL",
    "GG5",
    "GLC",
    "GOL",
    "HOH",
    "IOD",
    "IPA",
    "IPH",
    "MES",
    "MG8",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MXE",
    "MYR",
    "HN4",
    "NO3",
    "OCT",
    "P4C",
    "P4G",
    "P6G",
    "PEG",
    "PE4",
    "PG0",
    "PGO",
    "PG4",
    "PGE",
    "PGF",
    "PO4",
    "PTL",
    "SCN",
    "SGM",
    "SIN",
    "SIN",
    "SO4",
    "SRT",
    "TAM",
    "TAR",
    "TBR",
    "TCE",
    "TFA",
    "TFA",
    "TLA",
    "TMA",
    "TRS",
    "UNX",
)
NUCLEOTIDES = (
    'DA', 'DT', 'DC', 'DG', 'A', 'C', 'T', 'G', 'U'
)
MetaColumns = (
    # Taken from https://bioservices.readthedocs.io/en/main/_modules/bioservices/uniprot.html#UniProt
    # Names & Taxonomy ================================================
    "accession",
    "id",
    "gene_names",
    "gene_primary",
    "gene_synonym",
    "gene_oln",
    "gene_orf",
    "organism_name",
    "organism_id",
    "protein_name",
    "xref_proteomes",
    "lineage",
    "virus_hosts",
    # Sequences ========================================================
    "fragment",
    "sequence",
    "length",
    "mass",
    "organelle",
    "cc_alternative_products",
    "error_gmodel_pred",
    "cc_mass_spectrometry",
    "cc_polymorphism",
    "cc_rna_editing",
    "cc_sequence_caution",
    "ft_var_seq",
    "ft_variant",
    "ft_non_cons",
    "ft_non_std",
    "ft_non_ter",
    "ft_conflict",
    "ft_unsure",
    "sequence_version",
    # Family and Domains ========================================
    "ft_coiled",
    "ft_compbias",
    "cc_domain",
    "ft_domain",
    "ft_motif",
    "protein_families",
    "ft_region",
    "ft_repeat",
    "ft_zn_fing",
    # Function ===================================================
    "absorption",
    "ft_act_site",
    "cc_activity_regulation",
    "ft_binding",
    "ft_ca_bind",
    "cc_catalytic_activity",
    "cc_cofactor",
    "ft_dna_bind",
    "ec",
    "cc_function",
    "kinetics",
    "ft_metal",
    "ft_np_bind",
    "cc_pathway",
    "ph_dependence",
    "redox_potential",
    # 'rhea_id',
    "ft_site",
    "temp_dependence",
    # Gene Ontology ==================================
    "go",
    "go_p",
    "go_f",
    "go_c",
    "go_id",
    # Interaction ======================================
    "cc_interaction",
    "cc_subunit",
    # EXPRESSION =======================================
    "cc_developmental_stage",
    "cc_induction",
    "cc_tissue_specificity",
    # Publications
    "lit_pubmed_id",
    # Date of
    "date_created",
    "date_modified",
    "date_sequence_modified",
    "version",
    # Structure
    "structure_3d",
    "ft_strand",
    "ft_helix",
    "ft_turn",
    # Subcellular location
    "cc_subcellular_location",
    "ft_intramem",
    "ft_topo_dom",
    "ft_transmem",
    # Miscellaneous ==========================
    "annotation_score",
    "cc_caution",
    "comment_count",
    # "feature",
    "feature_count",
    "keyword",
    "keywordid",
    "cc_miscellaneous",
    "protein_existence",
    "tools",
    "reviewed",
    "uniparc_id",
    # Pathology
    "cc_allergen",
    "cc_biotechnology",
    "cc_disruption_phenotype",
    "cc_disease",
    "ft_mutagen",
    "cc_pharmaceutical",
    "cc_toxic_dose",
    # PTM / Processsing
    "ft_chain",
    "ft_crosslnk",
    "ft_disulfid",
    "ft_carbohyd",
    "ft_init_met",
    "ft_lipid",
    "ft_mod_res",
    "ft_peptide",
    "cc_ptm",
    "ft_propep",
    "ft_signal",
    "ft_transit",
    # not documented
    "xref_pdb",
)


@dataclass(frozen=True)
class _DumpNames:
    """
    Dataclass encapsulating names of files used for dumping data.
    """

    sequence: str = "sequence.tsv"
    meta: str = "meta.tsv"
    variables: str = "variables.tsv"
    segments_dir: str = "segments"
    structures_dir: str = "structures"
    structure_base_name: str = "structure"


@dataclass(frozen=True)
class _SeqNames:
    """
    Container holding names used within
    :attr:`lXtractor.core.Protein.ChainSequence._seqs`
    """

    seq1: str = "seq1"
    seq3: str = "seq3"
    enum: str = "numbering"
    map_canonical: str = "map_canonical"
    map_other: str = "map_other"
    map_aln: str = "map_aln"
    map_pdb: str = "map_pdb"
    map_ref: str = "map_ref"


@dataclass(frozen=True)
class _MetaNames:
    id: str = "id"
    name: str = "name"
    variables: str = "variables"
    structure_id: str = "structure_id"
    structure_chain_id: str = "structure_chain_id"
    category: str = "category"
    res_id: str = "res_id"
    res_name: str = "res_name"
    altloc: str = "altloc"


@dataclass(frozen=True)
class _ColNames:
    id: str = "ObjectID"
    parent_id: str = "ParentID"
    start: str = "Start"
    end: str = "End"


@dataclass(frozen=True)
class BondThresholds:
    """
    Holds covalent and non-covalent bond length distance thresholds,
    in angstroms.
    """

    covalent: Bounds = Bounds(1.2, 1.8)
    non_covalent: Bounds = Bounds(1.8, 5)


@dataclass(frozen=True)
class LigandConfig:
    """
    Config with parameters for ligand detection.
    """

    #: The distance thresholds for various bond types.
    bond_thresholds: BondThresholds = BondThresholds()
    #: The min number of a ligand's atoms.
    min_atoms: int = 5
    #: The min number of a structure's atoms forming at least bonds
    #: with a ligand.
    min_atom_connections: int = 5
    #: The min number of a structure's residues forming contact with a ligand.
    min_res_connections: int = 3


class AtomMark(IntFlag):
    """
    The atom categories. Some categories may be combined, e.g., LIGAND | PEP
    is another valid category denoting ligand peptide atoms.
    """
    #: Unknown atom.
    UNK: int = 1
    #: Solvent atom.
    SOLVENT: int = 2
    #: Ligand atom. If not combined with PEP, NUC, or CARB, this category
    #: denotes non-polymer (small molecule) single-residue ligands.
    LIGAND: int = 4
    #: Peptide polymer atoms.
    PEP: int = 8
    #: Nucleotide polymer atoms.
    NUC: int = 16
    #: Carbohydrate polymer atoms.
    CARB: int = 32


@dataclass(frozen=False)
class StructureConfig:
    """
    Structure configuration parameters. Needed to initialize
    :class:`lXtractor.core.structure.GenericStructure` objects.

    """
    #: A primary polymer type. "auto" will determine the primary polymer type
    #: as the one having the most atoms. Other valid values are "carbohydrate",
    #: "nucleotide", or "peptide". Abbreviations ("c", "n", or "p") are
    #: supported.
    primary_pol_type: str = "auto"
    #: Which polymer types can also be ligands.
    ligand_pol_types: tuple[str, ...] = ("c", "n", "p")
    #: The number of monomers to consider a polymeric entity a polymer.
    n_monomers: int = 2
    #: The ligand configuration parameters.
    ligand_config: LigandConfig = LigandConfig()
    #: The list of solvent three-letter codes.
    solvents: tuple[str, ...] = SOLVENTS
    #: Atom marks (types/categories).
    marks: AtomMark = AtomMark


DumpNames = _DumpNames()
SeqNames = _SeqNames()
MetaNames = _MetaNames()
ColNames = _ColNames()

if __name__ == "__main__":
    raise RuntimeError
