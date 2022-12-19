from collections import namedtuple
from dataclasses import dataclass

Separators = namedtuple(
    'Separators', ['list', 'chain', 'dom', 'uni_pdb', 'str', 'start_end']
)
Sep = Separators(',', ':', '::', '_', '--', '|')


@dataclass
class _DumpNames:
    """
    Dataclass encapsulating names of files used for dumping data.
    """

    sequence: str = 'sequence.tsv'
    meta: str = 'meta.tsv'
    variables: str = 'variables.tsv'
    pdist_base_dir: str = 'PDIST'
    pdist_base_name: str = 'pdist'
    segments_dir: str = 'segments'
    structures_dir: str = 'structures'
    structure_base_name: str = 'structure'


@dataclass
class _SeqNames:
    """
    Container holding names used within
    :attr:`lXtractor.core.Protein.ChainSequence._seqs`
    """

    seq1: str = 'seq1'
    seq3: str = 'seq3'
    enum: str = 'numbering'
    map_canonical: str = 'map_canonical'
    map_other: str = 'map_other'
    map_aln: str = 'map_aln'
    map_pdb: str = 'map_pdb'
    map_ref: str = 'map_ref'


@dataclass
class _MetaNames:
    id: str = 'id'
    name: str = 'name'
    variables: str = 'variables'
    pdb_id: str = 'pdb_id'
    pdb_chain: str = 'pdb_chain'
    category: str = 'category'


DumpNames = _DumpNames()
SeqNames = _SeqNames()
MetaNames = _MetaNames()

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
    'ft_coiled',
    'ft_compbias',
    'cc_domain',
    'ft_domain',
    'ft_motif',
    'protein_families',
    'ft_region',
    'ft_repeat',
    'ft_zn_fing',
    # Function ===================================================
    'absorption',
    'ft_act_site',
    'cc_activity_regulation',
    'ft_binding',
    'ft_ca_bind',
    'cc_catalytic_activity',
    'cc_cofactor',
    'ft_dna_bind',
    'ec',
    'cc_function',
    'kinetics',
    'ft_metal',
    'ft_np_bind',
    'cc_pathway',
    'ph_dependence',
    'redox_potential',
    # 'rhea_id',
    'ft_site',
    'temp_dependence',
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
    'cc_allergen',
    'cc_biotechnology',
    'cc_disruption_phenotype',
    'cc_disease',
    'ft_mutagen',
    'cc_pharmaceutical',
    'cc_toxic_dose',
    # PTM / Processsing
    'ft_chain',
    'ft_crosslnk',
    'ft_disulfid',
    'ft_carbohyd',
    'ft_init_met',
    'ft_lipid',
    'ft_mod_res',
    'ft_peptide',
    'cc_ptm',
    'ft_propep',
    'ft_signal',
    'ft_transit',
    # not documented
    'xref_pdb',
)

if __name__ == '__main__':
    raise RuntimeError
