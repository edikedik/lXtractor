import inspect
import typing as t
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqRecord, Seq
from Bio.PDB.Structure import Structure

SeqRec = SeqRecord.SeqRecord
Seq = Seq.Seq
T = t.TypeVar('T')
_Fetcher = t.Callable[[t.Iterable[str]], T]
_Getter = t.Callable[[T, t.Sequence[str]], t.Sequence[str]]
_Add_method = t.Callable[
    [t.Union[t.Iterable[SeqRec], Path], t.Iterable[SeqRec]],
    t.Tuple[t.Sequence[SeqRec], t.Sequence[SeqRec]]
]
_Align_method = t.Callable[
    [t.Iterable[SeqRec]], t.Sequence[SeqRec]
]

Separators = namedtuple('Separators', ['list', 'chain', 'dom', 'uni_pdb', 'str', 'start_end'])
Sep = Separators(',', ':', '::', '_', '--', '/')


class AminoAcidDict:
    """
    Complete and flexible amino acid dictionary, mapping between
    3->1 and 1->3-letter codes.

    >>> d = AminoAcidDict()
    >>> assert d['A'] == 'ALA'
    >>> assert d['ALA'] == 'A'
    >>> assert d['XXX'] == 'X'
    >>> assert d['X'] == 'UNK'

    """

    def __init__(
            self,
            aa1_unk: str = 'X',
            aa3_unk: str = 'UNK',
            any_unk: t.Optional[str] = None):
        """
        :param aa1_unk: unknown character when mapping 3->1
        :param aa3_unk: unknown character when mapping 1->3
        :param any_unk: unknown character when a key doesn't
            meet 1 or 3 length requirements.
        """
        self.aa1_unk = aa1_unk
        self.aa3_unk = aa3_unk
        self.any_unk = any_unk
        self._aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'THR': 'T', 'GLU': 'E',
            'GLH': 'e', 'ASP': 'D', 'ASH': 'd', 'PHE': 'F',
            'TRP': 'W', 'ILE': 'I', 'VAL': 'V', 'LEU': 'L',
            'LYS': 'K', 'LYN': 'k', 'MET': 'M', 'ASN': 'N',
            'GLN': 'Q', 'SER': 'S', 'ARG': 'R', 'TYR': 'Y',
            'TYD': 'y', 'HID': 'h', 'HIE': 'j', 'HIP': 'H',
            'HIS': 'H', 'PRO': 'P', 'GLY': 'G', 'A': 'ALA',
            'C': 'CYS', 'T': 'THR', 'E': 'GLU', 'e': 'GLH',
            'D': 'ASP', 'd': 'ASH', 'F': 'PHE', 'W': 'TRP',
            'I': 'ILE', 'V': 'VAL', 'L': 'LEU', 'K': 'LYS',
            'k': 'LYN', 'M': 'MET', 'N': 'ASN', 'Q': 'GLN',
            'S': 'SER', 'R': 'ARG', 'Y': 'TYR', 'y': 'TYD',
            'h': 'HID', 'j': 'HIE', 'H': 'HIS', 'P': 'PRO',
            'G': 'GLY'}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        """
        :return: unprotonated version of an amino acid code
        """
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H',
                'GLH': 'GLU', 'ASH': 'ASP', 'LYN': 'LYS',
                'TYD': 'TYR', 'HID': 'HIP', 'HIE': 'HIP'}

    @property
    def three_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 3-letter codes
        """
        return list(filter(lambda x: len(x) == 3, self._aa_dict))

    @property
    def one_letter_codes(self) -> t.List[str]:
        """
        :return: list of available 1-letter codes
        """
        return list(filter(lambda x: len(x) == 1, self._aa_dict))

    def __getitem__(self, item):
        if item in self._aa_dict:
            return self._aa_dict[item]
        if len(item) == 3:
            return self.aa1_unk
        elif len(item) == 1:
            return self.aa3_unk
        else:
            if self.any_unk is not None:
                return self.any_unk
            raise KeyError(
                f'Expected 3-sized or 1-sized item, '
                f'got {len(item)}-sized {item}')

    def __contains__(self, item):
        return item in self.aa_dict


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining basic interface any resource must provide.
    """

    def __init__(self, resource_path: t.Optional[Path],
                 resource_name: t.Optional[str]):
        self.name = resource_name
        self.path = resource_path

    @abstractmethod
    def read(self):
        """
        Read the resource using the :attr:`resource_path`
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        """
        Parse the read resource, so it's ready for usage.
        """
        raise NotImplementedError

    @abstractmethod
    def dump(self, path: Path):
        """
        Save the resource under the given `path`.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch(self, url: str):
        """
        Download the resource.
        """
        raise NotImplementedError


class AbstractVariable(metaclass=ABCMeta):
    """
    Abstract base class for variables.
    """

    __slots__ = ()

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (not isinstance(other, type(self)) or
                self.id == other.id)

    def __hash__(self):
        return hash(self.id)

    @property
    def id(self) -> str:
        """
        Variable identifier such that eval(x.id) produces another instance.
        """

        def parse_value(v):
            if isinstance(v, str):
                return f"\'{v}\'"
            return v

        init_params = inspect.signature(self.__init__).parameters
        args = ','.join(f'{k}={parse_value(v)}' for k, v in vars(self).items() if k in init_params)
        return f'{self.__class__.__name__}({args})'

    @property
    @abstractmethod
    def rtype(self) -> str:
        """
        A string such that ``eval(result_type)(result)`` converts the result
        into the correct type. For instance, "float".
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(
            self, obj: t.Union[Structure, SeqRec],
            mapping: t.Optional[t.Mapping[int, int]] = None
    ) -> t.Union[str, float]:
        """
        Calculate the variable. Each variable defines its own calculation
        strategy within this method.

        :param obj: An object used for variable's calculation.
        :param mapping: An optional mapping between an ``obj``'s sequence
            and alignment positions.
        :return: Calculation result.
        :raises: :class:`FailedCalculation` if the calculation fails.
        """
        raise NotImplementedError


class StructureVariable(AbstractVariable):
    """
    A type of variable whose :meth:`calculate` method requires protein structure.
    """

    @abstractmethod
    def calculate(self, obj: Structure, mapping: t.Optional[t.Mapping[int, int]] = None):
        raise NotImplementedError


class SequenceVariable(AbstractVariable):
    """
    A type of variable whose :meth:`calculate` method requires protein sequence.
    """

    @abstractmethod
    def calculate(self, obj: SeqRec, mapping: t.Optional[t.Mapping[int, int]] = None):
        raise NotImplementedError


class Variables(t.Dict):
    # TODO: inherit from UserDict instead
    """
    A subclass of :class:`dict` holding variables (:class:`AbstractVariable` subclasses).

    The keys are the :class:`AbstractVariable` subclasses' instances (since these are hashable objects),
    and values are calculation results.
    """

    @property
    def structure(self) -> t.Iterator[StructureVariable]:
        """
        :return: values that are :class:`StructureVariable` instances.
        """
        return filter(lambda v: isinstance(v, StructureVariable), self.keys())

    @property
    def sequence(self) -> t.Iterator[SequenceVariable]:
        """
        :return: values that are :class:`SequenceVariable` instances.
        """
        return filter(lambda v: isinstance(v, SequenceVariable), self.keys())


@dataclass
class _ProteinDumpNames:
    """
    Dataclass encapsulating names of files
    used for dumping data.
    """
    uniprot_seq: str = 'sequence.fasta'
    pdb_seq1: str = 'structure.fasta'
    pdb_seq3: str = 'structure.txt'
    pdb_structure: str = 'structure.pdb'
    meta: str = 'meta.tsv'
    variables: str = 'variables.tsv'
    aln_mapping: str = 'aln_mapping.tsv'
    uni_pdb_map: str = 'uni_pdb_map.tsv'
    pdist_base_dir: str = 'PDIST'
    pdist_base_name: str = 'pdist'
    uni_pdb_aln: str = 'uni_pdb_aln.fasta'
    domains_dir: str = 'domains'


ProteinDumpNames = _ProteinDumpNames()


@dataclass
class Segment:
    """
    Dataclass holding a definition of an arbitrary segment.

    Minimum data required is ``start`` and ``and``.
    It can be helpful to define both name of the segment and
    where it comes from (via ``parent_name``).

    A dictionary with arbitrary data resides within ``data``
    attribute.
    """
    start: int
    end: int
    name: t.Optional[str] = None
    parent_name: t.Optional[str] = None
    data: t.Optional[t.Dict[str, t.Any]] = field(
        default_factory=dict)

    @property
    def id(self) -> str:
        parent = f'(<-{self.parent_name})' if self.parent_name else ''
        return f'{self.name}{parent}:{self.start}-{self.end}'

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    def __iter__(self):
        return iter(range(self.start, self.end + 1))

    def __len__(self):
        if self.end is None or self.end is None or self.start >= self.end:
            return 0
        return self.end - self.start + 1

    def do_overlap(self, other: 'Segment') -> bool:
        """
        Check whether a segment overlaps with the other segment.
        Use :meth:`overlap` to find an actual overlap.

        :param other: other :class:`Segment` instance.
        :return: ``True`` if segments overlap and ``False`` otherwise.
        """
        return not (other.start > self.end or self.start > other.end)

    def overlap(self, other: 'Segment') -> t.Optional['Segment']:
        """
        If segments overlap, create a "child" segment with
        overlapping boundaries, ``name`` and a ``parent_name``
        of a given ``other``.

        :param other: other :class:`Segment` instance.
        :return: new :class:`Segment` instance.
        """

        if not self.do_overlap(other):
            return None

        return Segment(
            max(self.start, other.start),
            min(self.end, other.end),
            name=other.name,
            parent_name=other.parent_name,
            data=other.data)


class InitError(ValueError):
    """
    A broad category exception for problems with
    an object's initialization
    """
    pass


class MissingData(ValueError):
    pass


class AmbiguousData(ValueError):
    pass


class AmbiguousMapping(ValueError):
    pass


class NoOverlap(ValueError):
    pass


class FormatError(ValueError):
    pass


class FailedCalculation(RuntimeError):
    pass


class LengthMismatch(ValueError):
    pass


class OverlapError(ValueError):
    pass


class ParsingError(ValueError):
    pass


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
    'xref_pdb'
)

if __name__ == '__main__':
    raise RuntimeError
