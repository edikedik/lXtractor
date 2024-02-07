from __future__ import annotations

import operator as op
import typing as t
from collections import abc
from io import StringIO
from itertools import tee
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from more_itertools import chunked_even

from lXtractor.ext.base import ApiBase
from lXtractor.util import read_fasta, write_fasta
from lXtractor.util.io import fetch_text, fetch_chunks, fetch_urls

BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"
T = t.TypeVar("T")


def make_url(accessions: abc.Iterable[str], fmt: str, fields: str | None) -> str:
    params = {
        "format": fmt,
        "query": " OR ".join(map(lambda a: f"accession:{a}", accessions)),
    }
    if fmt == "tsv" and fields is not None:
        params["fields"] = fields

    return f"{BASE_URL}?{urlencode(params)}"


def url_getters() -> dict[str, abc.Callable[..., str]]:
    return {
        "sequences": lambda acc: make_url(acc, "fasta", None),
        "info": lambda acc, fields: make_url(acc, "tsv", fields),
    }


def _filter_existing(
    accessions: abc.Iterable[str], dir_: Path, fmt: str
) -> abc.Iterator[str]:
    existing = {p.stem for p in dir_.glob(f"*.{fmt}")}
    return filter(lambda x: x not in existing, accessions)


class UniProt(ApiBase):
    """
    An interface to UniProt fetching.

    :attr:`UniProt.url_getters` defines functions that construct a URL from
    provided arguments to fetch specific data. For instance, calling a URL
    getter for sequences in fasta format using a list of sequences will
    construct a valid URL for fetching the data.

    >>> uni = UniProt()
    >>> uni.url_getters['sequences'](['P00523', 'P12931'])
    'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=accession%3AP00523+OR+accession%3AP12931'

    These URLs are constructed dynamically within this class's methods, used
    to query UniProt, fetch and parse the data.

    """

    def __init__(
        self,
        chunk_size: int = 100,
        max_trials: int = 1,
        num_threads: int = 1,
        verbose: bool = False,
    ):
        """
        :param chunk_size: A number of IDs to join within a single URL and
            query simultaneously. Note that having invalid URL in a chunk
            invalidates all its IDs: they won't be fetched. For optimal
            performance, please filter your accessions carefully.
        :param max_trials: A maximum number of trials for fetching a single
            chunk. Makes sense to raise above ``1`` when the connection is
            unstable.
        :param num_threads: The number of threads to use for fetching chunks
            in parallel.
        :param verbose: Display progress bar via stdout.
        """
        super().__init__(url_getters(), max_trials, num_threads, verbose)
        self.chunk_size = chunk_size

    @t.overload
    def fetch_sequences(
        self, accessions, dir_, overwrite, callback: None
    ) -> abc.Iterator[tuple[str, str]]:
        ...

    @t.overload
    def fetch_sequences(
        self, accessions, dir_, overwrite, callback: abc.Callable[[tuple[str, str]], T]
    ) -> abc.Iterator[T]:
        ...

    def fetch_sequences(
        self,
        accessions: abc.Iterable[str],
        dir_: Path | None = None,
        overwrite: bool = False,
        callback: abc.Callable[[tuple[str, str]], T] | None = None,
    ) -> abc.Iterator[tuple[str, str]] | abc.Iterator[T]:
        """
        Fetch sequences in "fasta" format from UniProt.

        :param accessions: A list of valid accessions to fetch.
        :param dir_: A directory where individual sequence will be stored.
            If exists, will filter accessions before fetching unless `overwrite`
            is ``True``.
        :param overwrite: Overwrite existing sequences if they exist in `dir_`.
        :param callback: A function accepting a single sequence and returning
            anything else. Can be useful to convert sequences into, eg,
            :class:~lXtractor.chain.sequence.ChainSequence` (for this, pass
            :meth:~lXtractor.chain.sequence.ChainSequence.from_tuple` here).
        :return: An iterator over fetched sequences (or whatever ``callback``
            returns).
        """
        if dir_ is not None and not overwrite:
            accessions = _filter_existing(accessions, dir_, "fasta")
        chunks = map(tuple, chunked_even(accessions, self.chunk_size))
        fetched, missed = fetch_urls(
            self.url_getters["sequences"],
            chunks,
            "fasta",
            None,
            args_applier=lambda fn, args: fn(args),
            decode=True,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )
        text = "".join(map(op.itemgetter(1), fetched))
        seqs = read_fasta(StringIO(text), strip_id=False)

        if dir_ is not None:
            seqs, seqs_ = tee(seqs)
            for h, s in seqs_:
                acc = h.split("|")[1]
                write_fasta([(h, s)], dir_ / f"{acc}.fasta")

        if callback is not None:
            seqs = map(callback, seqs)

        return seqs

    def fetch_info(
        self,
        accessions: abc.Iterable[str],
        fields: str | None = None,
        as_df: bool = True,
    ) -> pd.DataFrame | list[str]:
        """
        Fetch information in tsv format from UniProt.

        :param accessions: A list of accessions to fetch the info for.
        :param fields: A comma-separated list of fields to fetch. If ``None``,
            default fields UniProt provides will be used.
        :param as_df: Convert fetched tables into pandas dataframes and join
            them. Otherwise, return raw text corresponding to each chunk of
            `accessions`.
        :return: A list of texts per chunk or a single data frame.
        """
        chunks = ((tuple(c), fields) for c in chunked_even(accessions, self.chunk_size))
        fetched, missed = fetch_urls(
            self.url_getters["info"],
            chunks,
            "tsv",
            None,
            decode=True,
            max_trials=self.max_trials,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )
        texts = map(op.itemgetter(1), fetched)
        if not as_df:
            return list(texts)
        return pd.concat(map(lambda x: pd.read_csv(StringIO(x), sep="\t"), texts))


def fetch_uniprot(
    acc: abc.Iterable[str],
    fmt: str = "fasta",
    chunk_size: int = 100,
    fields: str | None = None,
    **kwargs,
) -> str:
    """
    An interface to the UniProt's search.

    Base URL: `https://rest.uniprot.org/uniprotkb/stream <https://rest.uniprot.org/uniprotkb/stream>`_

    Available DB identifiers: See `bioservices <https://bioservices.readthedocs.io/en/main/_modules/bioservices/uniprot.html>`

    :param acc: an iterable over UniProt accessions.
    :param fmt: download format (e.g., "fasta", "gff", "tab", ...).
    :param chunk_size: how many accessions to download in a chunk.
    :param fields: if the ``fmt`` is "tsv", must be provided
        to specify which data columns to fetch.
    :param kwargs: passed to :func:`~lXtractor.util.io.fetch_chunks`.
    :return: the 'utf-8' encoded results as a single chunk of text.
    """

    def fetch_chunk(chunk: abc.Iterable[str]):
        full_url = make_url(chunk, fmt, fields)
        return fetch_text(full_url, decode=True)

    results = fetch_chunks(acc, fetch_chunk, chunk_size, **kwargs)

    return "".join(map(op.itemgetter(1), results))


if __name__ == "__main__":
    raise ValueError
