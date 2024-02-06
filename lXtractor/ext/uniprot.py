from __future__ import annotations

import operator as op
from collections import abc
from urllib.parse import urlencode

from lXtractor.util.io import fetch_text, fetch_chunks


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

    url = "https://rest.uniprot.org/uniprotkb/stream"

    def fetch_chunk(chunk: abc.Iterable[str]):
        params = {
            "format": fmt,
            "query": " OR ".join(map(lambda a: f"accession:{a}", chunk)),
        }
        if fmt == "tsv" and fields is not None:
            params["fields"] = fields
        return fetch_text(url, decode=True, params=urlencode(params).encode("utf-8"))

    results = fetch_chunks(acc, fetch_chunk, chunk_size, **kwargs)

    return "".join(map(op.itemgetter(1), results))


if __name__ == "__main__":
    raise ValueError
