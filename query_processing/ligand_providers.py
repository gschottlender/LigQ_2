from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from compound_processing.compound_helpers import LigandStore
from query_processing.results_tables import CompoundDatabaseProviderAdapter


class LigandSearchProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    def method_signature(self) -> dict:
        ...

    @abstractmethod
    def database_fingerprint(self, data_dir: Path) -> str:
        ...

    @abstractmethod
    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        ...

    def cache_method_signature(self) -> dict:
        return self.method_signature()

    def cache_coverage(self) -> tuple[float | None, float | None]:
        return None, None

    def with_cache_coverage(self, threshold_min: float | None, threshold_max: float | None):
        return self

    def filter_cached_results(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class CompoundDatabaseSearchProvider(LigandSearchProvider):
    def __init__(
        self,
        data_dir: Path,
        provider_name: str,
        target_base_name: str,
        search_representation: str = "morgan_1024_r2",
        search_metric: str = "tanimoto",
        search_threshold: float = 0.5,
        search_threshold_max: float | None = None,
        cluster_threshold: float = 0.8,
        search_per_iteration_topk: int = 1000,
        search_global_topk: int = 50000,
    ):
        self.data_dir = Path(data_dir)
        self._provider_name = str(provider_name)
        self.target_base_name = str(target_base_name)
        self.target_root = self.data_dir / "compound_data" / self.target_base_name
        self.compound_prefix = "ZINC" if self._provider_name == "zinc" else ""
        self.search_representation = search_representation
        self.search_metric = search_metric
        self.search_threshold = float(search_threshold)
        self.search_threshold_max = (
            float(search_threshold_max) if search_threshold_max is not None else None
        )
        self.cluster_threshold = float(cluster_threshold)
        self.search_per_iteration_topk = int(search_per_iteration_topk)
        self.search_global_topk = int(search_global_topk)

        pdb_chembl_root = self.data_dir / "compound_data" / "pdb_chembl"

        self.store_pdb_chembl = LigandStore(pdb_chembl_root)
        self.store_target = LigandStore(self.target_root)
        self.rep_pdb_chembl = self.store_pdb_chembl.load_representation("morgan_1024_r2")
        self.rep_target = self.store_target.load_representation("morgan_1024_r2")

        if self.search_representation == "morgan_1024_r2":
            search_rep_ref = self.rep_pdb_chembl
            search_rep_target = self.rep_target
        else:
            search_rep_ref = self.store_pdb_chembl.load_representation(self.search_representation)
            search_rep_target = self.store_target.load_representation(self.search_representation)

        self.adapter = CompoundDatabaseProviderAdapter(
            store_pdb_chembl=self.store_pdb_chembl,
            rep_pdb_chembl=self.rep_pdb_chembl,
            store_target=self.store_target,
            rep_target=self.rep_target,
            search_rep_ref=search_rep_ref,
            search_rep_target=search_rep_target,
            search_metric=self.search_metric,
            search_threshold=self.search_threshold,
            search_threshold_max=self.search_threshold_max,
            cluster_threshold=self.cluster_threshold,
            search_per_iteration_topk=self.search_per_iteration_topk,
            search_global_topk=self.search_global_topk,
            compound_prefix=self.compound_prefix,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def method_signature(self) -> dict:
        return {
            "provider": self.provider_name,
            "search_representation": self.search_representation,
            "search_metric": self.search_metric,
            "search_threshold": self.search_threshold,
            "search_threshold_max": self.search_threshold_max,
        }

    def cache_method_signature(self) -> dict:
        # Cache identity is intentionally defined only by the user-facing
        # search method. Thresholds are tracked as cache coverage in the
        # manifest so wider caches can serve stricter queries later on.
        return {
            "provider": self.provider_name,
            "search_representation": self.search_representation,
            "search_metric": self.search_metric,
        }

    @property
    def score_column(self) -> str:
        return "tanimoto" if self.search_metric == "tanimoto" else "similarity"

    def cache_coverage(self) -> tuple[float | None, float | None]:
        return self.search_threshold, self.search_threshold_max

    def with_cache_coverage(
        self,
        threshold_min: float | None,
        threshold_max: float | None,
    ) -> "CompoundDatabaseSearchProvider":
        return CompoundDatabaseSearchProvider(
            data_dir=self.data_dir,
            provider_name=self.provider_name,
            target_base_name=self.target_base_name,
            search_representation=self.search_representation,
            search_metric=self.search_metric,
            search_threshold=self.search_threshold if threshold_min is None else threshold_min,
            search_threshold_max=threshold_max,
            cluster_threshold=self.cluster_threshold,
            search_per_iteration_topk=self.search_per_iteration_topk,
            search_global_topk=self.search_global_topk,
        )

    def filter_cached_results(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.score_column not in df.columns:
            return df
        filtered = df[df[self.score_column] >= self.search_threshold]
        if self.search_threshold_max is not None:
            filtered = filtered[filtered[self.score_column] <= self.search_threshold_max]
        return filtered.reset_index(drop=True)

    def database_fingerprint(self, data_dir: Path) -> str:
        data_dir = Path(data_dir)
        target_root = data_dir / "compound_data" / self.target_base_name
        reps_root = target_root / "reps"
        meta_path = reps_root / f"{self.search_representation}.meta.json"

        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Representation metadata not found for '{self.search_representation}' at: {meta_path}"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        rep_data_path = reps_root / meta["file"]
        ligands_path = target_root / "ligands.parquet"
        def _fp(path: Path) -> dict:
            st = path.stat()
            return {
                "path": str(path.relative_to(data_dir)),
                "size": int(st.st_size),
            }

        files = [meta_path, rep_data_path, ligands_path]

        payload = {
            "provider": self.provider_name,
            "search_representation": self.search_representation,
            "files": [_fp(p) for p in files],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        return self.adapter.compute_for_protein(prot=prot, known_binding=known_binding)


class ZincLigandSearchProvider(CompoundDatabaseSearchProvider):
    def __init__(
        self,
        data_dir: Path,
        search_representation: str = "morgan_1024_r2",
        search_metric: str = "tanimoto",
        zinc_search_threshold: float = 0.5,
        zinc_search_threshold_max: float | None = None,
        cluster_threshold: float = 0.8,
        zinc_per_iteration_topk: int = 1000,
        zinc_global_topk: int = 50000,
    ):
        super().__init__(
            data_dir=data_dir,
            provider_name="zinc",
            target_base_name="zinc",
            search_representation=search_representation,
            search_metric=search_metric,
            search_threshold=zinc_search_threshold,
            search_threshold_max=zinc_search_threshold_max,
            cluster_threshold=cluster_threshold,
            search_per_iteration_topk=zinc_per_iteration_topk,
            search_global_topk=zinc_global_topk,
        )


def build_provider(
    provider_name: str,
    data_dir: Path,
    search_representation: str,
    search_metric: str,
    search_threshold: float,
    search_threshold_max: float | None,
    cluster_threshold: float,
    search_per_iteration_topk: int,
    search_global_topk: int,
) -> LigandSearchProvider:
    if provider_name == "zinc":
        return ZincLigandSearchProvider(
            data_dir=data_dir,
            search_representation=search_representation,
            search_metric=search_metric,
            zinc_search_threshold=search_threshold,
            zinc_search_threshold_max=search_threshold_max,
            cluster_threshold=cluster_threshold,
            zinc_per_iteration_topk=search_per_iteration_topk,
            zinc_global_topk=search_global_topk,
        )
    target_root = Path(data_dir) / "compound_data" / provider_name
    if not target_root.exists():
        raise ValueError(
            f"Unknown ligand provider '{provider_name}'. Expected compound database at: {target_root}"
        )
    return CompoundDatabaseSearchProvider(
        data_dir=data_dir,
        provider_name=provider_name,
        target_base_name=provider_name,
        search_representation=search_representation,
        search_metric=search_metric,
        search_threshold=search_threshold,
        search_threshold_max=search_threshold_max,
        cluster_threshold=cluster_threshold,
        search_per_iteration_topk=search_per_iteration_topk,
        search_global_topk=search_global_topk,
    )
