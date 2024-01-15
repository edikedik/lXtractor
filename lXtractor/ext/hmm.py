"""
Wrappers around PyHMMer for convenient annotation of domains and families.
"""
from __future__ import annotations

import gzip
import logging
import typing as t
from collections import abc
from itertools import count
from pathlib import Path
from shutil import rmtree

import pandas as pd
from more_itertools import peekable, split_at
from pyhmmer.easel import (
    DigitalSequence,
    TextSequence,
    DigitalSequenceBlock,
    TextMSA,
    DigitalMSA,
    Alphabet,
)
from pyhmmer.plan7 import (
    HMM,
    HMMFile,
    Pipeline,
    TopHits,
    Alignment,
    Domain,
    TraceAligner,
    Builder,
    Background,
)

from lXtractor.chain import ChainSequence, ChainStructure, Chain
from lXtractor.core import Alignment as lXAlignment
from lXtractor.core.base import AbstractResource
from lXtractor.core.exceptions import MissingData
from lXtractor.util import fetch_to_file

LOGGER = logging.getLogger(__name__)

HMM_DEFAULT_NAME = "HMM"
PFAM_HMM_NAME = "Pfam-A.hmm.gz"
PFAM_DAT_NAME = "Pfam-A.hmm.dat.gz"
PFAM_HMM_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
PFAM_DAT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz"
)
PFAM_LOC = Path(__file__).parent.parent / "resources" / "Pfam"
_ChainT: t.TypeAlias = Chain | ChainStructure | ChainSequence
_SeqT: t.TypeAlias = _ChainT | str | tuple[str, str] | DigitalSequence
_HmmInpT: t.TypeAlias = HMM | HMMFile | Path | str
CT = t.TypeVar("CT", bound=t.Union[ChainSequence, ChainStructure, Chain])


# TODO: verify non-domain HMM types (e.g., Motif, Family) yield valid Domain hits
def iter_hmm(hmm: _HmmInpT) -> abc.Generator[HMM]:
    """
    Iterate over HMM models.

    :param hmm: A path to an HMM file, opened ``HMMFile`` or a stream.
    :return: An iterator over individual HMM models.
    """
    match hmm:
        case HMMFile():
            yield from hmm
        case HMM():
            yield hmm
        case _:
            with HMMFile(hmm) as f:
                yield from f


def _enumerate_numbering(
    a: Alignment,
) -> abc.Generator[tuple[int | None, int | None], None, None]:
    hmm_pool, seq_pool = count(a.hmm_from), count(a.target_from)
    for hmm_c, seq_c in zip(a.hmm_sequence, a.target_sequence):
        hmm_i = None if hmm_c == "." else next(hmm_pool)
        seq_i = None if seq_c == "-" else next(seq_pool)
        yield seq_i, hmm_i


def _get_alphabet(alphabet: Alphabet | str):
    if isinstance(alphabet, str):
        if alphabet.lower() == "amino":
            alphabet = Alphabet.amino()
        elif alphabet.lower() == "dna":
            alphabet = Alphabet.dna()
        elif alphabet.lower() == "rna":
            alphabet = Alphabet.rna()
        else:
            raise ValueError(f"Invalid alphabet type {alphabet}.")
    return alphabet


def digitize_seq(obj: t.Any, alphabet: Alphabet | str = "amino") -> DigitalSequence:
    """
    :param obj: A `Chain*`-type object or string or a tuple of (name, _seq).
        A sequence of this object must be compatible with the alphabet of
        the HMM model.
    :param alphabet: An alphabet type the sequence corresponds to. Can be an
        initialized PyHMMer alphabet or a string "amino", "dna", or "rna".
    :return: A digitized sequence compatible with PyHMMer.
    """
    alphabet = _get_alphabet(alphabet)

    match obj:
        case DigitalSequence():
            return obj
        case TextSequence():
            return obj.digitize(alphabet)
        case str():
            _id = str(hash(obj))
            accession, name, text = _id, _id, obj
        case [str(), str()]:
            accession, name, text = obj[0], obj[0], obj[1]
        case ChainSequence():
            accession, name, text = obj.id, obj.name, obj.seq1
        case ChainStructure() | Chain():
            accession, name, text = obj.id, obj.id, obj.seq.seq1
        case _:
            raise TypeError(f"Unsupported sequence type {type(obj)}")
    return TextSequence(
        sequence=text,
        name=bytes(name, encoding="utf-8"),
        accession=bytes(accession, encoding="utf-8"),
    ).digitize(alphabet)


class PyHMMer:
    """
    A basis pyhmmer interface aimed at domain extraction.
    It works with a single hmm model and pipeline instance.

    `The original documentation <https://pyhmmer.readthedocs.io/en/stable/>`.
    """

    def __init__(self, hmm: _HmmInpT, **kwargs):
        """
        :param hmm: An :class:`HMMFile` handle or path as string or `Path`
            object to a file containing a single HMM model. In case of multiple
            models, only the first one will be taken
        :param kwargs: Passed to :class:`Pipeline`. The `alphabet` argument
            is derived from the supplied `hmm`.
        """
        try:
            #: HMM instance
            self.hmm = next(iter_hmm(hmm))
        except (StopIteration, EOFError) as e:
            raise MissingData(f"Invalid input {hmm}") from e
        #: Pipeline to use for HMM searches
        self.pipeline: Pipeline = self.init_pipeline(**kwargs)
        #: Hits resulting from the most recent HMM search
        self.hits_: TopHits | None = None

    @classmethod
    def from_hmm_collection(cls, hmm: _HmmInpT, **kwargs) -> abc.Generator[t.Self]:
        """
        Split HMM collection and initialize a :class:`PyHMMer` instance from
        each HMM model.

        :param hmm: A path to HMM file, opened HMMFile handle, or parsed HMM.
        :param kwargs: Passed to the class constructor.
        :return: A generator over :class:`PyHMMer` instances created from the
            provided HMM models.
        """
        yield from (cls(inp, **kwargs) for inp in iter_hmm(hmm))

    @classmethod
    def from_msa(
        cls,
        msa: abc.Iterable[tuple[str, str] | str | _ChainT] | lXAlignment,
        name: str | bytes,
        alphabet: Alphabet | str,
        **kwargs,
    ) -> t.Self:
        """
        Create a :class:`PyHMMer` instance from a multiple sequence alignment.

        :param msa: An iterable over sequences.
        :param name: The HMM model's name.
        :param alphabet: An alphabet to use to build the HMM model.
            See :func:`digitize_seq` for available options.
        :param kwargs: Passed to :class:`DigitalMSA` of ``PyHMMer`` that serves
            as the basis for creating an HMM model.
        :return: A new :class:`PyHMMer` instance initialized with the HMM model
            built here.
        """
        if isinstance(name, str):
            name = bytes(name, "utf-8")
        alphabet = _get_alphabet(alphabet)
        msa_d = DigitalMSA(
            alphabet,
            name,
            sequences=list(map(digitize_seq, msa)),
            **kwargs,
        )
        builder = Builder(alphabet)
        background = Background(alphabet)
        hmm, _, _ = builder.build_msa(msa_d, background)
        return cls(hmm)

    def init_pipeline(self, **kwargs) -> Pipeline:
        """
        :param kwargs: Passed to :class:`Pipeline` during initialization.
        :return: Initialized pipeline, also saved to :attr:`pipeline`.
        """
        self.pipeline = Pipeline(self.hmm.alphabet, **kwargs)
        return self.pipeline

    def convert_seq(self, obj: t.Any) -> DigitalSequence:
        """
        :param obj: A `Chain*`-type object or string or a tuple of (name, _seq).
            A sequence of this object must be compatible with the alphabet of
            the HMM model.
        :return: A digitized sequence compatible with PyHMMer.
        """
        return digitize_seq(obj, self.hmm.alphabet)

    def _convert_to_seq_block(self, seqs: abc.Iterable[_SeqT]) -> DigitalSequenceBlock:
        return DigitalSequenceBlock(self.hmm.alphabet, map(self.convert_seq, seqs))

    def search(self, seqs: abc.Iterable[_SeqT]) -> TopHits:
        """
        Run the :attr:`pipeline` to search for :attr:`hmm`.

        :param seqs: Iterable over digital sequences or objects accepted by
            :meth:`convert_seq`.
        :return: Top hits resulting from the search.
        """
        if self.pipeline is None:
            self.init_pipeline()

        seqs_block = self._convert_to_seq_block(seqs)
        self.hits_ = self.pipeline.search_hmm(self.hmm, seqs_block)
        return self.hits_

    def align(self, seqs: abc.Iterable[_SeqT]) -> TextMSA:
        """
        Align sequences to a profile.

        :param seqs: Sequences to align.
        :return: :class:`TextMSA` with aligned sequences.
        """
        block = self._convert_to_seq_block(seqs)
        aligner = TraceAligner()
        traces = aligner.compute_traces(self.hmm, block)
        msa = aligner.align_traces(self.hmm, block, traces)
        assert isinstance(msa, TextMSA), "Unexpected MSA type returned"
        return msa

    def annotate(
        self,
        objs: abc.Iterable[_ChainT] | _ChainT,
        new_map_name: str | None = None,
        min_score: float | None = None,
        min_size: int | None = None,
        min_cov_hmm: float | None = None,
        min_cov_seq: float | None = None,
        domain_filter: abc.Callable[[Domain], bool] | None = None,
        **kwargs,
    ) -> abc.Generator[CT, None, None]:
        """
        Annotate provided objects by hits resulting from the HMM search.

        An annotation is the creation of a child object via :meth:`spawn_child`
        method (e.g., :meth:`lXtractor.core.chain.ChainSequence.spawn_child`).

        :param objs: A single one or an iterable over `Chain*`-type objects.
        :param new_map_name: A name for a child
            :class:`ChainSequence <lXtractor.core.chain.ChainSequence` to hold
            the mapping to the hmm numbering.
        :param min_score: Min hit score.
        :param min_size: Min hit size.
        :param min_cov_hmm: Min HMM model coverage -- a fraction of
            mapped / total nodes.
        :param min_cov_seq: Min coverage of a sequence by the HMM model
            nodes -- a fraction of mapped nodes to the sequence's length.
        :param domain_filter: A callable to filter domain hits.
        :param kwargs: Passed to the `spawn_child` method.
            **Hint:** if you don't want to keep spawned children,
            pass ``keep=False`` here.
        :return: A generator over spawned children yielded sequentially
            for each input object and valid domain hit.
        """

        def accept_domain(d: Domain, cov_hmm: float, cov_seq: float) -> bool:
            acc_cov_hmm = min_cov_hmm is None or cov_hmm >= min_cov_hmm
            acc_cov_seq = min_cov_seq is None or cov_seq >= min_cov_seq
            acc_score = min_score is None or d.score >= min_score
            acc_size = min_size is None or (
                d.alignment.target_to - d.alignment.target_from >= min_size
            )
            return all([acc_cov_hmm, acc_cov_seq, acc_score, acc_size])

        if isinstance(objs, _ChainT):
            objs = [objs]
        else:
            peeking = peekable(objs)
            fst = peeking.peek(False)
            if not fst:
                raise MissingData("No sequences provided")
            if not isinstance(fst, _ChainT):
                raise TypeError(f"Unsupported type {fst}")

        if new_map_name is None:
            try:
                new_map_name = self.hmm.accession.decode("utf-8").split(".")[0]
            except ValueError:
                try:
                    new_map_name = (
                        self.hmm.name.decode("utr-8")
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                except ValueError:
                    LOGGER.warning(
                        "`new_map_name` was not provided, and neither `accession` "
                        "nor `name` attribute exist: falling back to default "
                        f"name: {HMM_DEFAULT_NAME}"
                    )
                    new_map_name = HMM_DEFAULT_NAME

        objs_by_id: dict[str, CT] = {s.id: s for s in objs}

        self.search(objs_by_id.values())

        for hit in self.hits_:
            obj = objs_by_id[hit.accession.decode("utf-8")]

            for dom_i, dom in enumerate(hit.domains, start=1):
                aln = dom.alignment

                # compute coverages
                # HMM: - 1 2 3 - 4 - 5
                # SEQ: A - A A A A - A

                # HMM node numbers falling onto valid sequence elements
                # => - 2 3 - 4 5
                num = [hmm_i for seq_i, hmm_i in _enumerate_numbering(aln) if seq_i]

                # n = the number of valid HMM nodes covered by _seq
                # => 2 3 4 5 => 4
                n = sum(1 for x in num if x is not None)

                # SEQ coverage 4 / 6
                cov_seq = n / len(num)
                # HMM coverage 4 / M
                cov_hmm = n / self.hmm.M

                if not accept_domain(dom, cov_hmm, cov_seq):
                    continue
                if domain_filter and not domain_filter(dom):
                    continue

                name = f"{new_map_name}_{dom_i}"
                offset = obj.start - 1
                sub = obj.spawn_child(
                    aln.target_from + offset,
                    aln.target_to + offset,
                    name,
                    **kwargs,
                )

                seq = sub.seq if isinstance(obj, (Chain, ChainStructure)) else sub
                seq.add_seq(new_map_name, num)
                seq.meta[f"{new_map_name}_pvalue"] = dom.pvalue
                seq.meta[f"{new_map_name}_score"] = dom.score
                seq.meta[f"{new_map_name}_bias"] = dom.bias
                seq.meta[f"{new_map_name}_cov_seq"] = str(cov_seq)
                seq.meta[f"{new_map_name}_cov_hmm"] = str(cov_hmm)
                yield sub


class Pfam(AbstractResource):
    """
    A minimalistic Pfam interface.

        * :meth:`fetch` fetches Pfam raw HMM models and associated metadata.
        * :meth:`parse` prepares these data for later usage and stores
            to the filesystem.
        * :meth:`read` loads parsed files.

    Parsed Pfam data is represented as a Pandas DataFrame accessible via
    :meth:`df` with columns: "ID", "Accession", "Description", "Category",
    and "HMM". Each row corresponds to a single model from Pfam-A collection
    and associated metadata taken from the Pfam-A.dat file. HMM models are
    wrapped into a :class:`PyHMMer` instance.

    For quick access to a single HMM model parsed into :class:`PyHMMer`,
    use ``Pfam()[hmm_id]``.

    """

    def __init__(
        self,
        resource_path: Path = PFAM_LOC,
        resource_name: str = "Pfam",
    ):
        super().__init__(resource_path, resource_name)
        self._df = None

    @property
    def df(self) -> pd.DataFrame | None:
        """
        :return: Parsed Pfam if :meth:`read` or :meth:`parse` were called.
            Otherwise, returns ``None``.
        """
        return self._df

    @property
    def dat_columns(self) -> tuple[str, ...]:
        return "ID", "Accession", "Description", "Category"

    @df.setter
    def df(self, value: pd.DataFrame):
        self._validate_frame(value)
        self._df = value

    def __getitem__(self, item: str) -> PyHMMer:
        path = self.path / "parsed" / "hmm" / f"{item}.hmm.gz"
        try:
            return PyHMMer(path)
        except FileNotFoundError:
            raise KeyError(f"No HMM with ID {item} found in {path}")

    def fetch(
        self,
        url_hmm: str = PFAM_HMM_URL,
        url_dat: str = PFAM_DAT_URL,
    ) -> tuple[Path, Path]:
        """
        Fetch Pfam-A data from InterPro.

        :param url_hmm: URL to "Pfam-A.hmm.gz".
        :param url_dat: URL to "Pfam-A.hmm.dat.gz"
        :return: A pair of filepaths for fetched HMM and dat files.
        """
        base = self.path / "raw"
        base.mkdir(exist_ok=True, parents=True)
        return (
            fetch_to_file(url_hmm, base / PFAM_HMM_NAME),
            fetch_to_file(url_dat, base / PFAM_DAT_NAME),
        )

    def parse(
        self,
        dump: bool = True,
        rm_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Parse fetched raw data into a single pandas :class:`DataFrame`.

        :param dump: Dump parsed files to :attr:`path` / "raw" dir.
        :param rm_raw: Clean up the raw data once parsing is done.
        :return: A parsed Pfam :class:`DataFrame`. See the class's docs for
            a list of columns.
        """
        path_hmm = self.path / "raw" / PFAM_HMM_NAME
        path_dat = self.path / "raw" / PFAM_DAT_NAME
        if not path_hmm.exists():
            raise FileNotFoundError(f"Missing raw HMM data in {path_hmm}")
        if not path_dat.exists():
            raise FileNotFoundError(f"Missing raw dat file in {path_dat}")
        df_hmm = self._parse_hmm(path_hmm)
        df_dat = self._parse_dat(path_dat)
        self._df = df_dat.merge(df_hmm, on="Accession")
        if dump:
            self.dump()
        if rm_raw:
            self.clean(raw=True)
        return self._df

    def read(
        self,
        path: Path | None = None,
        accessions: abc.Container[str] | None = None,
        categories: abc.Container[str] | None = None,
        hmm: bool = True,
    ) -> pd.DataFrame:
        """
        Read parsed Pfam data.

        First it reads the "dat" file and filters to relevant accessions and/or
        categories. Then, if `hmm` is ``True``, it loads each model and wraps
        into an :class:`PyHMMer` instance. Otherwise, it loads the HMM metadata.
        One can explore and filter these data, then load the desired HMM models
        via :meth:`load_hmm`.

        :param path: A path to the dir with layout similar to what :meth:`dump`
            creates.
        :param accessions: A list of Pfam accessions following the ".", e.g.,
            ``["PF00069", ]``.
        :param categories: A list of Pfam categories to filter the accessions to.
        :param hmm: Load HMM models.
        :return: A parsed Pfam :class:`DataFrame`.
        """
        base = path or self.path / "parsed"
        df = pd.read_csv(base / "dat.csv")
        if accessions:
            df = df[df["Accession"].isin(accessions)]
        if categories:
            df = df[df["Category"].isin(categories)]
        if hmm:
            df = self.load_hmm(df)
        self._df = df
        return df

    def load_hmm(
        self, df: pd.DataFrame | None = None, path: Path | None = None
    ) -> pd.DataFrame:
        """
        Load HMM models according to accessions in passed `df` and create a
        column "PyHMMer" with loaded models.

        :param df: A :class:`DataFrame` having all the :meth:`dat_columns.
        :param path: A custom path to the parsed data with an "hmm" subdir.
        :return: A copy of the original :class:`DataFrame` with loaded models.
        """
        base = path or self.path / "parsed"
        df = self._df if df is None else df
        if df is None:
            raise MissingData("No parsed data. Use `read` or `parse` first.")
        df = df.copy()
        df["PyHMMer"] = [
            PyHMMer(base / "hmm" / f"{acc}.hmm.gz") for acc in df["Accession"]
        ]
        return df

    def dump(self, path: Path | None = None) -> Path:
        """
        Store parsed data to the filesystem.

        This function will store the HMM metadata to attr:`path` / "parsed"
        / "dat.csv" and separate gzip-compressed HMM models into :attr:`path`
        / "parsed" / "hmm".

        :param path: Use this path instead of the :attr:`path` as a base dir.
        :return: The path :attr:`path` / "parsed".
        """
        base = path or self.path / "parsed"
        (base / "hmm").mkdir(exist_ok=True, parents=True)
        if self._df is None:
            raise MissingData("Missing parsed data to dump")

        for _, row in self._df.iterrows():
            hmm_path = base / "hmm" / f"{row.Accession}.hmm.gz"
            with gzip.open(hmm_path, "wb") as f:
                row.HMM.hmm.write(f)

        self._df[["ID", "Accession", "Description", "Category"]].to_csv(
            base / "dat.csv", index=False
        )
        return base

    def clean(self, raw: bool = True, parsed: bool = False) -> None:
        """
        Remove Pfam data. If `raw` and `parsed` are both ``False``, removes
        the :attr:`path` with all stored data.

        :param raw: Remove raw fetched files.
        :param parsed: Remove parsed files.
        :return: Nothing.
        """
        if raw:
            rmtree(self.path / "raw")
        elif parsed:
            rmtree(self.path / "parsed")
        else:
            rmtree(self.path)

    def _parse_dat(self, path: Path) -> pd.DataFrame:
        def wrap_chunk(xs: list[str]):
            _id, _acc, _desc, _, _type = map(lambda x: x.split("   ")[-1], xs[:5])
            _acc = _acc.split(".")[0]
            return _id, _acc, _desc, _type

        with gzip.open(path, "rt") as f:
            lines = filter(bool, map(lambda x: x.rstrip(), f))
            chunks = filter(bool, split_at(lines, lambda x: x.startswith("# ")))
            return pd.DataFrame(
                map(wrap_chunk, chunks),
                columns=list(self.dat_columns),
            )

    @staticmethod
    def _parse_hmm(path: Path) -> pd.DataFrame:
        df = pd.DataFrame({"HMM": list(map(PyHMMer, iter_hmm(path)))})
        df["Accession"] = df["HMM"].map(
            lambda x: x.hmm.accession.decode("utf-8").split(".")[0]
        )
        return df

    def _validate_frame(self, df: pd.DataFrame):
        for c in self.dat_columns:
            if c not in df.columns:
                raise MissingData(f"Missing required column {c}")


if __name__ == "__main__":
    raise RuntimeError
