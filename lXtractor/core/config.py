"""
A module encompassing various settings of lXtractor objects.
"""
from __future__ import annotations

import json
from collections import UserDict, abc
from contextlib import contextmanager
from copy import deepcopy
from enum import IntFlag
from pathlib import Path

STRUCTURE_EXT = (".cif", ".pdb", ".mmtf")
STRUCTURE_FMT = tuple(x[1:] for x in STRUCTURE_EXT)

_RESOURCES = Path(__file__).parent.parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "default_config.json"
_USER_CONFIG_PATH = _RESOURCES / "user_config.json"

MetaColumns = (
    # TODO: move to docs
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


class Config(UserDict):
    """
    A configuration management class.

    This class facilitates the loading and saving of configuration settings,
    with a user-specified configuration overriding the default settings.

    :param default_config_path: The path to the default config file. This is a
        reference default settings, which can be used to reset user settings
        if needed.
    :param user_config_path: The path to the user configuration file. This file
        is stored internally and can be modified by a user to provide permanent
        settings.

    Loading and mofifying the config:

    >>> cfg = Config()
    >>> list(cfg.keys())[:2]
    ['bonds', 'colnames']
    >>> cfg['bonds']['non_covalent_upper']
    5.0
    >>> cfg['bonds']['non_covalent_upper'] = 6

    Equivalently, one can update the config by a local JSON file or dict:

    >>> cfg.update_with({'bonds': {'non_covalent_upper': 4}})
    >>> assert cfg['bonds']['non_covalent_upper'] == 4

    The changes can be stored internally and loaded automatically in the future:

    >>> cfg.save()
    >>> cfg = Config()
    >>> assert cfg['bonds']['non_covalent_upper'] == 4

    To restore default settings:

    >>> cfg.reset_to_defaults()
    >>> cfg.clear_user_config()
    """

    def __init__(
        self,
        default_config_path: str | Path = _DEFAULT_CONFIG_PATH,
        user_config_path: str | Path = _USER_CONFIG_PATH,
    ):
        self.default_config_path = Path(default_config_path)
        self.user_config_path = Path(user_config_path)

        super().__init__()
        self.reload()

    @contextmanager
    def temporary_namespace(self):
        """
        A context manager for a temporary config namespace.

        Within this context, changes to the config are allowed, but will be
        reverted back to the original config once the context is exited.

        Example:

        >>> cfg = Config()
        >>> with cfg.temporary_namespace():
        ...     cfg['bonds']['non_covalent_upper'] = 10
        ...     # Do some stuff with the temporary config...
        ... # Config is reverted back to original state here
        >>> assert cfg['bonds']['non_covalent_upper'] != 10
        """
        original_config = deepcopy(self.data)
        try:
            yield self  # This allows access to the Config object within the context
        finally:
            self.data = original_config  # Revert config back to original state

    def reload(self):
        """Load the configuration from files."""
        # Load true default config
        with self.default_config_path.open("r") as f:
            self.data.update(json.load(f))

        # Update with default user config
        with self.user_config_path.open("r") as f:
            user_default_config = json.load(f)
        self.update_with(user_default_config)

    def save(self, user_config_path: str | Path = _USER_CONFIG_PATH):
        """
        Save the current configuration. By default, will store the configuration
        internally. This stored configuration will be loaded automatically on
        top of the default configuration.

        :param user_config_path: The path where to save the user configuration file.
        :raises ValueError: If the user config path is not provided.
        """
        with Path(user_config_path).open("w") as f:
            json.dump(self.data, f, indent=4)

    def reset_to_defaults(self):
        """Reset the configuration to the default settings."""
        self.data.clear()
        with self.default_config_path.open("r") as f:
            self.data.update(json.load(f))

    def clear_user_config(self):
        """Clear the contents of the locally stored user config file."""
        with self.user_config_path.open("w") as f:
            json.dump({}, f, indent=4)

    def update_with(
        self, other: abc.Mapping[str, abc.Mapping[str, list[str] | str | float]] | Path
    ):
        if isinstance(other, Path):
            with other.open() as f:
                self.update(json.load(f))
        else:
            for k, v in other.items():
                if k in self:
                    self[k].update(v)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


DefaultConfig = Config()


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


if __name__ == "__main__":
    raise RuntimeError
EMPTY_ALTLOC = ("", " ", ".")
