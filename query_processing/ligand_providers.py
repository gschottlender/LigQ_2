from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from compound_processing.compound_helpers import LigandStore
from query_processing.results_tables import ZincProviderAdapter


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


class ZincLigandSearchProvider(LigandSearchProvider):
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
        self.data_dir = Path(data_dir)
        self.search_representation = search_representation
        self.search_metric = search_metric
        self.zinc_search_threshold = float(zinc_search_threshold)
        self.zinc_search_threshold_max = (
            float(zinc_search_threshold_max) if zinc_search_threshold_max is not None else None
        )
        self.cluster_threshold = float(cluster_threshold)
        self.zinc_per_iteration_topk = int(zinc_per_iteration_topk)
        self.zinc_global_topk = int(zinc_global_topk)

        pdb_chembl_root = self.data_dir / "compound_data" / "pdb_chembl"
        zinc_root = self.data_dir / "compound_data" / "zinc"

        self.store_pdb_chembl = LigandStore(pdb_chembl_root)
        self.store_zinc = LigandStore(zinc_root)
        self.rep_pdb_chembl = self.store_pdb_chembl.load_representation("morgan_1024_r2")
        self.rep_zinc = self.store_zinc.load_representation("morgan_1024_r2")

        if self.search_representation == "morgan_1024_r2":
            search_rep_ref = self.rep_pdb_chembl
            search_rep_zinc = self.rep_zinc
        else:
            search_rep_ref = self.store_pdb_chembl.load_representation(self.search_representation)
            search_rep_zinc = self.store_zinc.load_representation(self.search_representation)

        self.adapter = ZincProviderAdapter(
            store_pdb_chembl=self.store_pdb_chembl,
            rep_pdb_chembl=self.rep_pdb_chembl,
            store_zinc=self.store_zinc,
            rep_zinc=self.rep_zinc,
            search_rep_ref=search_rep_ref,
            search_rep_zinc=search_rep_zinc,
            search_metric=self.search_metric,
            zinc_search_threshold=self.zinc_search_threshold,
            zinc_search_threshold_max=self.zinc_search_threshold_max,
            cluster_threshold=self.cluster_threshold,
            zinc_per_iteration_topk=self.zinc_per_iteration_topk,
            zinc_global_topk=self.zinc_global_topk,
        )

    @property
    def provider_name(self) -> str:
        return "zinc"

    def method_signature(self) -> dict:
        return {
            "provider": self.provider_name,
            "search_representation": self.search_representation,
            "search_metric": self.search_metric,
            "zinc_search_threshold": self.zinc_search_threshold,
            "zinc_search_threshold_max": self.zinc_search_threshold_max,
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
        return self.zinc_search_threshold, self.zinc_search_threshold_max

    def with_cache_coverage(
        self,
        threshold_min: float | None,
        threshold_max: float | None,
    ) -> "ZincLigandSearchProvider":
        return ZincLigandSearchProvider(
            data_dir=self.data_dir,
            search_representation=self.search_representation,
            search_metric=self.search_metric,
            zinc_search_threshold=self.zinc_search_threshold if threshold_min is None else threshold_min,
            zinc_search_threshold_max=threshold_max,
            cluster_threshold=self.cluster_threshold,
            zinc_per_iteration_topk=self.zinc_per_iteration_topk,
            zinc_global_topk=self.zinc_global_topk,
        )

    def filter_cached_results(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.score_column not in df.columns:
            return df
        filtered = df[df[self.score_column] >= self.zinc_search_threshold]
        if self.zinc_search_threshold_max is not None:
            filtered = filtered[filtered[self.score_column] <= self.zinc_search_threshold_max]
        return filtered.reset_index(drop=True)

    def database_fingerprint(self, data_dir: Path) -> str:
        data_dir = Path(data_dir)
        zinc_root = data_dir / "compound_data" / "zinc"
        reps_root = zinc_root / "reps"
        meta_path = reps_root / f"{self.search_representation}.meta.json"

        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Representation metadata not found for '{self.search_representation}' at: {meta_path}"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        rep_data_path = reps_root / meta["file"]
        ligands_path = zinc_root / "ligands.parquet"
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


def build_provider(
    provider_name: str,
    data_dir: Path,
    search_representation: str,
    search_metric: str,
    zinc_search_threshold: float,
    zinc_search_threshold_max: float | None,
    cluster_threshold: float,
    zinc_per_iteration_topk: int,
    zinc_global_topk: int,
) -> LigandSearchProvider:
    if provider_name == "zinc":
        return ZincLigandSearchProvider(
            data_dir=data_dir,
            search_representation=search_representation,
            search_metric=search_metric,
            zinc_search_threshold=zinc_search_threshold,
            zinc_search_threshold_max=zinc_search_threshold_max,
            cluster_threshold=cluster_threshold,
            zinc_per_iteration_topk=zinc_per_iteration_topk,
            zinc_global_topk=zinc_global_topk,
        )
    raise ValueError(f"Unknown ligand provider: {provider_name}")
