from __future__ import annotations

import operator as op
from collections import abc
from urllib.parse import urlencode

from lXtractor.ext.base import ApiBase
from lXtractor.util.io import fetch_text, fetch_chunks

BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"


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
        "info": lambda acc, fields: make_url(acc, "tsv", None),
    }


class UniProt(ApiBase):
    pass


def fetch_uniprot(
    acc: abc.Iterable[str],
    fmt: str = "fasta",
    chunk_size: int = 100,
    fields: str | None = None,
    **kwargs,
) -> str:
    """
    An interface to the UniProt's search.

    Base URL: https://rest.uniprot.org/uniprotkb/stream

    Available DB identifiers: https://bioservices.readthedocs.io/en/main/_modules/bioservices/uniprot.html

    Will use :func:`fetch_chunks lXtractor.util.io.fetch_chunks` internally.

    :param acc: an iterable over UniProt accessions.
    :param fmt: download format (e.g., "fasta", "gff", "tab", ...).
    :param chunk_size: how many accessions to download in a chunk.
    :param fields: if the ``fmt`` is "tsv", must be provided
        to specify which data columns to fetch.
    :param kwargs: passed to :func:`fetch_chunks lXtractor.util.io.fetch_chunks`.
    :return: the 'utf-8' encoded results as a single chunk of text.
    """

    def fetch_chunk(chunk: abc.Iterable[str]):
        full_url = make_url(chunk, fmt, fields)
        return fetch_text(full_url, decode=True)

    results = fetch_chunks(acc, fetch_chunk, chunk_size, **kwargs)

    return "".join(map(op.itemgetter(1), results))


if __name__ == "__main__":
    raise ValueError
