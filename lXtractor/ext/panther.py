from __future__ import annotations

import json
from collections import abc
from urllib.parse import urlencode

import pandas as pd

import lXtractor.util as util

__all__ = ("fetch_orthologs_info",)


def fetch_orthologs_info(
    accessions: abc.Iterable[str],
    organism: int | str = 9606,
    target_organisms: abc.Iterable[int | str] | None = None,
    ortholog_type: str = "LDO",
    to_df: bool = True,
    **kwargs,
) -> dict | pd.DataFrame:
    url = "http://pantherdb.org/services/oai/pantherdb/ortholog/matchortho"

    params = {"geneInputList": ",".join(accessions), "organism": organism}

    if target_organisms:
        params["targetOrganism"] = ",".join(map(str, target_organisms))

    if ortholog_type:
        params["orthologType"] = ortholog_type

    res = json.loads(
        util.fetch_text(
            url, decode=True, params=urlencode(params).encode("utf-8"), **kwargs
        )
    )

    if to_df:
        try:
            res = pd.DataFrame(res["search"]["mapping"]["mapped"])
        except KeyError:
            raise KeyError(
                "Failed to convert to DataFrame. Missing required keys in the "
                'retrieved input. Required keys are: "search", "mapping", and '
                '"mapped". Returning a raw JSON output.'
            )
        finally:
            return res


if __name__ == "__main__":
    raise RuntimeError
