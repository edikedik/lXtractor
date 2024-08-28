from collections import abc
from itertools import dropwhile
from pathlib import Path
from tempfile import NamedTemporaryFile

import biotite.structure as bst
import numpy as np
import pandas as pd
from biotite.structure.io.pdb import PDBFile
from more_itertools import unzip

from lXtractor.core import GenericStructure, ProteinStructure
from lXtractor.util import load_structure, save_structure, run_sp

__all__ = ["dssp_to_df", "dssp_run", "dssp_set_ss_annotation"]

DSSP_TO_SS8 = {"H": "H", "G": "G", "I": "I", "E": "E", "B": "B", "T": "T", "S": "S"}
DSSP_TO_SS3 = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}
DSSP_COLUMNS = (
    "res_id",
    "chain_id",
    "res_name",
    "structure",
    "ss8",
    "ss3",
    "bp1",
    "bp2",
    "acc",
    "n_h_o_1",
    "o_h_n_1",
    "n_h_o_2",
    "o_h_n_2",
    "tco",
    "kappa",
    "alpha",
    "phi",
    "psi",
    "x_ca",
    "y_ca",
    "z_ca",
)


def _parse_dssp_line(line):
    ss = line[16]
    return {
        "res_id": int(line[0:5].strip()),
        "chain_id": line[11],
        "res_name": line[13],
        "structure": line[16:25].strip(),
        "bp1": int(line[25:29]),
        "bp2": int(line[29:33]),
        "acc": int(line[34:38]),
        "n_h_o_1": line[39:50].strip(),
        "o_h_n_1": line[50:61].strip(),
        "n_h_o_2": line[61:72].strip(),
        "o_h_n_2": line[72:83].strip(),
        "tco": float(line[83:91]),
        "kappa": float(line[91:97]),
        "alpha": float(line[97:103]),
        "phi": float(line[103:109]),
        "psi": float(line[109:115]),
        "x_ca": float(line[115:122]),
        "y_ca": float(line[122:129]),
        "z_ca": float(line[129:136]),
        "ss8": DSSP_TO_SS8.get(ss, "C"),
        "ss3": DSSP_TO_SS3.get(ss, "C"),
    }


def dssp_to_df(output: str | Path) -> pd.DataFrame:
    """
    Parse "classical" DSSP output and convert it into a pandas dataframe.

    :param output: Full output as a str or a path to an output file.
    :return: A dataframe the same columns as in the DSSP output plust two
        additional columns: "ss8" and "ss3" with secondary structure
        designations (original 8-state and converted 3-state).
    """

    def parse(lines: abc.Iterable[str]) -> abc.Iterable[str]:
        lines = filter(lambda x: bool(x.strip()), lines)
        lines = dropwhile(lambda x: not x.startswith("  #  RESIDUE AA "), lines)
        next(lines)  # consume header
        yield from map(_parse_dssp_line, lines)

    if isinstance(output, str):
        lines = output.split("\n")
    elif isinstance(output, Path):
        lines = output.read_text().splitlines()
    else:
        raise TypeError(f"Invalid output type {type(output)}. Must be str or Path.")

    df = pd.DataFrame(parse(lines))
    df = df[df["chain_id"] != " "]  # remove polymer line breaks
    return df[list(DSSP_COLUMNS)]


def dssp_set_ss_annotation(a: bst.AtomArray, df: pd.DataFrame) -> None:
    """
    Set secondary structure annotations for a given atom array.
    Modifies the provided array in-place to incorporate "ss8" and "ss3"
    annotation categories with ``np.nan`` designating missing annotations.

    :param a: Atom array.
    :param df: Dataframe with parsed DSSP output.
    :return: Nothing.

    .. seealso::
        :func:`dssp_to_df` :func:`dssp_run`
    """
    sub = df[["res_id", "chain_id", "ss3", "ss8"]]
    ss_map = {
        (res_id, chain_id): (ss3, ss8)
        for res_id, chain_id, ss3, ss8 in sub.itertuples(index=False)
    }
    starts = bst.get_residue_starts(a)
    annotations = (
        ss_map.get((atom.res_id, atom.chain_id), (np.nan, np.nan)) for atom in a[starts]
    )
    ss3, ss8 = map(
        lambda x: bst.spread_residue_wise(a, np.array(list(x))), unzip(annotations)
    )
    a.set_annotation("ss3", ss3)
    a.set_annotation("ss8", ss8)


def dssp_run(
    structure: GenericStructure | ProteinStructure | bst.AtomArray | Path,
    exec_name: str = "mkdssp",
    set_ss_annotation: bool = False,
) -> pd.DataFrame:
    """
    Run DSSP and on a given structure and parse the ouptut.

    :param structure: Parsed structure or an atom array or path to a structure.
    :param exec_name: Name of the DSSP executable.
    :param set_ss_annotation: Set secondary structure annotations. Only works
        if `structure` is not a ``Path``.
    :return: Parsed DSSP output.

    .. seealso::
        :func:`dssp_set_ss_annotation` :func:`dssp_to_df`
    """
    if isinstance(structure, (GenericStructure, ProteinStructure)):
        a = structure.array
    elif isinstance(structure, Path):
        a = load_structure(structure)
    elif isinstance(structure, bst.AtomArray):
        a = structure
    else:
        raise TypeError(f"Invalid structure type {type(structure)}")

    # New DSSP versions work primarily with mmCIF. However, mmCIF written by
    # biotite is missing some important header info, and DSSP doesn't work with
    # it. Hence, we write the array in PDB format and prepend a dummy header
    # so DSSP is happy with the PURITY of the PDB format and the input not
    # being GARBAGE (https://github.com/PDB-REDO/dssp/issues/10)
    with NamedTemporaryFile("w", suffix=".pdb") as tmp:
        file = PDBFile()
        file.set_structure(a)
        tmp.write("HEADER empty\n")
        file.write(tmp)
        tmp.seek(0)
        output = run_sp(f"{exec_name} --output-format dssp {tmp.name}").stdout
        df = dssp_to_df(output)

    if set_ss_annotation and not isinstance(structure, Path):
        dssp_set_ss_annotation(a, df)

    return df


if __name__ == "__main__":
    raise RuntimeError
