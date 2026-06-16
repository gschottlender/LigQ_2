from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from compound_processing.bsi_search import (
    BSI_DEFAULT_MAX_KNOWN_LIGANDS,
    BSI_MODEL_SUBDIR,
    BSI_REPRESENTATION,
    BSIModelRegistry,
    get_bsi_ligands,
)
from compound_processing.compound_helpers import LigandStore
from query_processing.results_tables import CompoundDatabaseProviderAdapter


def _file_fingerprint(path: Path, data_dir: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path.relative_to(data_dir)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _representation_files(root: Path, rep_name: str) -> list[Path]:
    reps_root = root / "reps"
    meta_path = reps_root / f"{rep_name}.meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Representation metadata not found for '{rep_name}' at: {meta_path}"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return [meta_path, reps_root / meta["file"]]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


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
        search_global_topk: int = 10000,
        search_device: str = "auto",
        search_q_batch_size: int | None = None,
        search_target_chunk_size: int | None = None,
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
        self.search_device = search_device
        self.search_q_batch_size = search_q_batch_size
        self.search_target_chunk_size = search_target_chunk_size

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
            search_device=self.search_device,
            search_q_batch_size=self.search_q_batch_size,
            search_target_chunk_size=self.search_target_chunk_size,
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
            search_device=self.search_device,
            search_q_batch_size=self.search_q_batch_size,
            search_target_chunk_size=self.search_target_chunk_size,
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
        pdb_chembl_root = data_dir / "compound_data" / "pdb_chembl"

        rep_names = ["morgan_1024_r2"]
        if self.search_representation != "morgan_1024_r2":
            rep_names.append(self.search_representation)

        files = [
            target_root / "ligands.parquet",
            pdb_chembl_root / "ligands.parquet",
            data_dir / "results_databases" / "known_binding_data.parquet",
        ]
        for rep_name in rep_names:
            files.extend(_representation_files(target_root, rep_name))
            files.extend(_representation_files(pdb_chembl_root, rep_name))

        files = _dedupe_paths(files)

        payload = {
            "provider": self.provider_name,
            "search_representation": self.search_representation,
            "files": [_file_fingerprint(path, data_dir) for path in files],
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
        search_threshold: float = 0.5,
        search_threshold_max: float | None = None,
        cluster_threshold: float = 0.8,
        search_per_iteration_topk: int = 1000,
        search_global_topk: int = 10000,
        *,
        zinc_search_threshold: float | None = None,
        zinc_search_threshold_max: float | None = None,
        zinc_per_iteration_topk: int | None = None,
        zinc_global_topk: int | None = None,
        search_device: str = "auto",
        search_q_batch_size: int | None = None,
        search_target_chunk_size: int | None = None,
    ):
        if zinc_search_threshold is not None:
            search_threshold = zinc_search_threshold
        if zinc_search_threshold_max is not None:
            search_threshold_max = zinc_search_threshold_max
        if zinc_per_iteration_topk is not None:
            search_per_iteration_topk = zinc_per_iteration_topk
        if zinc_global_topk is not None:
            search_global_topk = zinc_global_topk
        super().__init__(
            data_dir=data_dir,
            provider_name="zinc",
            target_base_name="zinc",
            search_representation=search_representation,
            search_metric=search_metric,
            search_threshold=search_threshold,
            search_threshold_max=search_threshold_max,
            cluster_threshold=cluster_threshold,
            search_per_iteration_topk=search_per_iteration_topk,
            search_global_topk=search_global_topk,
            search_device=search_device,
            search_q_batch_size=search_q_batch_size,
            search_target_chunk_size=search_target_chunk_size,
        )


class BSILigandSearchProvider(LigandSearchProvider):
    def __init__(
        self,
        data_dir: Path,
        target_base_name: str,
        bsi_threshold: float = 0.5,
        cluster_threshold: float = 0.8,
        search_per_iteration_topk: int = 1000,
        search_global_topk: int = 10000,
        search_device: str = "auto",
        search_target_chunk_size: int | None = None,
        bsi_model_batch_size: int = 65536,
        bsi_max_known_ligands: int = BSI_DEFAULT_MAX_KNOWN_LIGANDS,
    ):
        self.data_dir = Path(data_dir)
        self.target_base_name = str(target_base_name)
        self._provider_name = f"{self.target_base_name}_bsi"
        self.bsi_threshold = float(bsi_threshold)
        self.cluster_threshold = float(cluster_threshold)
        self.search_per_iteration_topk = int(search_per_iteration_topk)
        self.search_global_topk = int(search_global_topk)
        self.search_device = search_device
        self.search_target_chunk_size = search_target_chunk_size
        self.bsi_model_batch_size = int(bsi_model_batch_size)
        self.bsi_max_known_ligands = int(bsi_max_known_ligands)
        self.compound_prefix = "ZINC" if self.target_base_name == "zinc" else ""

        pdb_chembl_root = self.data_dir / "compound_data" / "pdb_chembl"
        target_root = self.data_dir / "compound_data" / self.target_base_name
        models_root = self.data_dir / BSI_MODEL_SUBDIR
        protein_domains_path = self.data_dir / "results_databases" / "protein_domains.parquet"
        if not protein_domains_path.is_file():
            raise FileNotFoundError(f"Protein domains table not found: {protein_domains_path}")

        self.store_pdb_chembl = LigandStore(pdb_chembl_root)
        self.store_target = LigandStore(target_root)
        self.rep_pdb_chembl = self.store_pdb_chembl.load_representation(BSI_REPRESENTATION)
        self.rep_target = self.store_target.load_representation(BSI_REPRESENTATION)
        if self.rep_pdb_chembl.dim != 1024 or self.rep_target.dim != 1024:
            raise ValueError("BSI requires morgan_1024_r2 fingerprints with 1024 bits.")

        self.model_registry = BSIModelRegistry(models_root)
        self.protein_domains = pd.read_parquet(protein_domains_path)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def score_column(self) -> str:
        return "bsi_score"

    def method_signature(self) -> dict:
        return {
            "provider": self.provider_name,
            "target_provider": self.target_base_name,
            "search_representation": BSI_REPRESENTATION,
            "search_metric": "bsi",
            "bsi_threshold": self.bsi_threshold,
        }

    def cache_method_signature(self) -> dict:
        return {
            "provider": self.provider_name,
            "target_provider": self.target_base_name,
            "search_representation": BSI_REPRESENTATION,
            "search_metric": "bsi",
            "bsi_model": "mpg_1024",
            "bsi_max_known_ligands": self.bsi_max_known_ligands,
        }

    def cache_coverage(self) -> tuple[float | None, float | None]:
        return self.bsi_threshold, None

    def with_cache_coverage(
        self,
        threshold_min: float | None,
        threshold_max: float | None,
    ) -> "BSILigandSearchProvider":
        return BSILigandSearchProvider(
            data_dir=self.data_dir,
            target_base_name=self.target_base_name,
            bsi_threshold=self.bsi_threshold if threshold_min is None else threshold_min,
            cluster_threshold=self.cluster_threshold,
            search_per_iteration_topk=self.search_per_iteration_topk,
            search_global_topk=self.search_global_topk,
            search_device=self.search_device,
            search_target_chunk_size=self.search_target_chunk_size,
            bsi_model_batch_size=self.bsi_model_batch_size,
            bsi_max_known_ligands=self.bsi_max_known_ligands,
        )

    def filter_cached_results(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.score_column not in df.columns:
            return df
        return df[df[self.score_column] >= self.bsi_threshold].reset_index(drop=True)

    def database_fingerprint(self, data_dir: Path) -> str:
        data_dir = Path(data_dir)
        target_root = data_dir / "compound_data" / self.target_base_name
        pdb_chembl_root = data_dir / "compound_data" / "pdb_chembl"

        files = [
            target_root / "ligands.parquet",
            pdb_chembl_root / "ligands.parquet",
            data_dir / "results_databases" / "known_binding_data.parquet",
            data_dir / "results_databases" / "protein_domains.parquet",
        ]
        files.extend(_representation_files(target_root, BSI_REPRESENTATION))
        files.extend(_representation_files(pdb_chembl_root, BSI_REPRESENTATION))
        files = _dedupe_paths(files)

        payload = {
            "provider": self.provider_name,
            "target_provider": self.target_base_name,
            "search_representation": BSI_REPRESENTATION,
            "bsi_models": self.model_registry.fingerprint(),
            "files": [],
        }
        for path in files:
            payload["files"].append(_file_fingerprint(path, data_dir))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        return get_bsi_ligands(
            prot=prot,
            known_binding=known_binding,
            protein_domains=self.protein_domains,
            model_registry=self.model_registry,
            store_pdb_chembl=self.store_pdb_chembl,
            rep_pdb_chembl=self.rep_pdb_chembl,
            store_target=self.store_target,
            rep_target=self.rep_target,
            max_queries=self.bsi_max_known_ligands,
            cluster_threshold=self.cluster_threshold,
            bsi_threshold=self.bsi_threshold,
            search_device=self.search_device,
            search_target_chunk_size=self.search_target_chunk_size,
            bsi_model_batch_size=self.bsi_model_batch_size,
            search_per_iteration_topk=self.search_per_iteration_topk,
            search_global_topk=self.search_global_topk,
            compound_prefix=self.compound_prefix,
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
    use_bsi: bool = False,
    bsi_threshold: float = 0.5,
    search_device: str = "auto",
    search_q_batch_size: int | None = None,
    search_target_chunk_size: int | None = None,
    bsi_model_batch_size: int = 65536,
    bsi_max_known_ligands: int = BSI_DEFAULT_MAX_KNOWN_LIGANDS,
) -> LigandSearchProvider:
    if use_bsi:
        target_root = Path(data_dir) / "compound_data" / provider_name
        if provider_name != "zinc" and not target_root.exists():
            raise ValueError(
                f"Unknown ligand provider '{provider_name}'. Expected compound database at: {target_root}"
            )
        return BSILigandSearchProvider(
            data_dir=data_dir,
            target_base_name=provider_name,
            bsi_threshold=bsi_threshold,
            cluster_threshold=cluster_threshold,
            search_per_iteration_topk=search_per_iteration_topk,
            search_global_topk=search_global_topk,
            search_device=search_device,
            search_target_chunk_size=search_target_chunk_size,
            bsi_model_batch_size=bsi_model_batch_size,
            bsi_max_known_ligands=bsi_max_known_ligands,
        )

    if provider_name == "zinc":
        return ZincLigandSearchProvider(
            data_dir=data_dir,
            search_representation=search_representation,
            search_metric=search_metric,
            search_threshold=search_threshold,
            search_threshold_max=search_threshold_max,
            cluster_threshold=cluster_threshold,
            search_per_iteration_topk=search_per_iteration_topk,
            search_global_topk=search_global_topk,
            search_device=search_device,
            search_q_batch_size=search_q_batch_size,
            search_target_chunk_size=search_target_chunk_size,
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
        search_device=search_device,
        search_q_batch_size=search_q_batch_size,
        search_target_chunk_size=search_target_chunk_size,
    )
