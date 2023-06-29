import typing as t

from .base import LigandVariable, MappingT, AggFns
from .util import atom_mask, residue_mask, _agg_dist
from ..core import Ligand
from ..core.exceptions import InitError

__all__ = ("LigandDist",)


class LigandDist(LigandVariable):
    def __init__(
        self, p: int, a: str | None = None, la: str | None = None, agg: str = "min"
    ):
        if agg not in AggFns:
            raise InitError(
                f"Wrong agg_lig {agg}. " f"Available aggregators: {list(AggFns)}"
            )

        self.p = p
        self.a = a
        self.la = la
        self.agg = agg

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(self, obj: Ligand, mapping: MappingT | None = None) -> float:
        parent_mask = (
            atom_mask(self.p, self.a, obj.parent.array, mapping)
            if self.a is not None
            else residue_mask(self.p, obj.parent.array, mapping)
        )
        if self.la is not None:
            lig_mask = atom_mask(obj.res_id, self.la, obj.array, None)
            obj_array = obj.array[lig_mask]
        else:
            obj_array = obj.array

        return _agg_dist(obj_array, obj.parent.array[parent_mask], AggFns[self.agg])


if __name__ == "__main__":
    raise RuntimeError
