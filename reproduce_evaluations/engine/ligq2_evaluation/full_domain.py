"""Build the complete shared-Pfam protein source for a benchmark target."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DomainSource:
    target: str
    query_uniprot_ids: tuple[str, ...]
    pfam_ids: tuple[str, ...]
    source_uniprot_ids: tuple[str, ...]
    n_domain_proteins: int
    n_excluded_benchmark_proteins: int


def build_full_domain_sources(
    targets: pd.DataFrame,
    binding: pd.DataFrame,
    selected_targets: set[str],
) -> dict[str, DomainSource]:
    """Select every ligand-bearing protein sharing a target Pfam.

    The historical neighbor benchmark excluded all benchmark-target UniProt
    IDs from candidate proteins. Apply the same exclusion here. Sources are
    sorted by UniProt ID so that the deterministic MaxMin initial candidate
    does not depend on row order in a Parquet file.
    """

    required_targets = {"target", "uniprot_id"}
    required_binding = {"uniprot_id", "pfam_id", "chem_comp_id"}
    if not required_targets.issubset(targets.columns):
        raise ValueError(f"Target table lacks {sorted(required_targets - set(targets.columns))}")
    if not required_binding.issubset(binding.columns):
        raise ValueError(f"Binding table lacks {sorted(required_binding - set(binding.columns))}")

    target_table = targets[["target", "uniprot_id"]].dropna().astype(str)
    excluded = set(target_table["uniprot_id"])
    annotated = binding[["uniprot_id", "pfam_id"]].dropna().astype(str).drop_duplicates()
    annotated = annotated[annotated["pfam_id"].ne("nan")]
    proteins_by_pfam = annotated.groupby("pfam_id")["uniprot_id"].agg(set).to_dict()
    pfams_by_protein = annotated.groupby("uniprot_id")["pfam_id"].agg(set).to_dict()
    ligand_bearing = set(binding.loc[binding["chem_comp_id"].notna(), "uniprot_id"].astype(str))

    result: dict[str, DomainSource] = {}
    for target in sorted(selected_targets):
        query_ids = tuple(sorted(set(target_table.loc[target_table["target"].eq(target), "uniprot_id"])))
        if not query_ids:
            raise ValueError(f"No UniProt ID is available for target {target}")
        pfams = tuple(sorted(set().union(*(pfams_by_protein.get(uid, set()) for uid in query_ids))))
        domain_proteins = set().union(*(proteins_by_pfam[pfam] for pfam in pfams)) if pfams else set()
        sources = tuple(sorted((domain_proteins - excluded) & ligand_bearing))
        result[target] = DomainSource(
            target=target,
            query_uniprot_ids=query_ids,
            pfam_ids=pfams,
            source_uniprot_ids=sources,
            n_domain_proteins=len(domain_proteins),
            n_excluded_benchmark_proteins=len(domain_proteins & excluded),
        )
    return result


def slice_binding_for_target(binding: pd.DataFrame, source: DomainSource) -> pd.DataFrame:
    """Retain every row needed for the unchanged split, pool, and seed logic."""

    query_rows = binding["uniprot_id"].isin(source.query_uniprot_ids)
    # The historical code stringifies missing Pfam values before forming its
    # background blacklist. Keep those rows if the query has a missing Pfam.
    query_pfams_for_background = set(binding.loc[query_rows, "pfam_id"].astype(str))
    pfam_rows = binding["pfam_id"].astype(str).isin(query_pfams_for_background)
    seed_rows = binding["uniprot_id"].isin(source.source_uniprot_ids)
    return binding.loc[query_rows | pfam_rows | seed_rows].copy()
