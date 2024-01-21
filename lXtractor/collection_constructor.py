import typing as t
import json
from pathlib import Path

from lXtractor.collection import (
    ChainCollection,
    StructureCollection,
    SequenceCollection,
)
from lXtractor.core.config import Config
from lXtractor.core.exceptions import MissingData, FormatError

_RESOURCES = Path(__file__).parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCES / "collection_config.json"
_USER_CONFIG_PATH = _RESOURCES / "collection_user_config.json"
_CT = t.TypeVar("_CT", SequenceCollection, StructureCollection, ChainCollection)
_CTA: t.TypeAlias = SequenceCollection | StructureCollection | ChainCollection


def _collection_type_from_source(source: t.Any) -> tuple[t.Type[_CTA], ...]:
    match source:
        case (_, None, None) | "UniProt":
            return (SequenceCollection,)
        case (None, _, None) | "PDB" | "AF2":
            return SequenceCollection, StructureCollection
        case (_, _, _) | "SIFTS":
            return SequenceCollection, ChainCollection
        case _:
            raise FormatError(
                f"Failed to determine collection type from source {source}"
            )


def _to_concrete_collection(desc: str) -> t.Type[_CT]:
    match desc[:3].lower():
        case "cha":
            return ChainCollection
        case "seq":
            return SequenceCollection
        case "str":
            return StructureCollection
        case _:
            raise NameError(f"Cannot determine collection from parameter {desc}")


class ConstructorConfig(Config):
    def __init__(
        self,
        default_config_path: str | Path = _DEFAULT_CONFIG_PATH,
        user_config_path: str | Path = _USER_CONFIG_PATH,
        **kwargs,
    ):
        self.provided_settings = kwargs
        super().__init__(default_config_path, user_config_path)

    def reload(self):
        """
        Reload the configuration from files and initially
        :attr:`provided_settings`
        """
        super().reload()
        self.update_with(self.provided_settings)

    def save(self, user_config_path: str | Path = _USER_CONFIG_PATH):
        super().save(user_config_path)

    @classmethod
    def list_fields(cls) -> list[str]:
        with _DEFAULT_CONFIG_PATH.open("r") as f:
            return list(json.load(f))

    def list_missing_fields(self) -> list[str]:
        return [k for k, v in self.data.items() if v is None]

    def validate(self) -> None:
        none_keys = ", ".join(self.list_missing_fields())
        if none_keys:
            raise MissingData(f"Missing values for required keys: {none_keys}")


class CollectionConstructor:
    def __init__(self, config: ConstructorConfig):
        self.config = config
        self._setup_paths()
        self.collection = self._setup_collection()
        self.config.validate()

    def _setup_collection(self) -> _CT:
        possible_types = _collection_type_from_source(self.config["source"])
        provided_type = _to_concrete_collection(self.config["collection_type"])
        if provided_type not in possible_types:
            raise ValueError(
                f"Collection type {provided_type} is not possible to create from "
                f"source {self.config['source']}."
            )
        coll_name = provided_type.__name__.removesuffix("Collection").lower()

        if self.config["collection_name"] == "auto":
            self.config["collection_name"] = coll_name
        coll_path = (
            self.config["output_dir"] / f"{self.config['collection_name']}.sqlite"
        )
        return provided_type(coll_path)

    def _setup_paths(self):
        for k in ("output_dir", "str_dir", "seq_dir"):
            self.config[k] = Path(self.config[k])
            self.config[k].mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise RuntimeError
