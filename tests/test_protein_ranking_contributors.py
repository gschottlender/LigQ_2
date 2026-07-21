from __future__ import annotations

import pandas as pd

from query_processing.query_processing_functions import _process_single_query


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qseqid": "query-1",
                "sseqid": "P1",
                "search_type": "sequence",
                "bitscore": 200.0,
                "evalue": 1e-40,
            },
            {
                "qseqid": "query-1",
                "sseqid": "P2",
                "search_type": "nearest_k",
                "bitscore": 150.0,
                "evalue": 1e-20,
            },
            {
                "qseqid": "query-1",
                "sseqid": "P3",
                "search_type": "domain",
                "best_domain_score": 80.0,
                "best_domain_evalue": 1e-10,
            },
            {
                "qseqid": "query-1",
                "sseqid": "P4",
                "search_type": "domain",
                "best_domain_score": 50.0,
                "best_domain_evalue": 1e-5,
            },
        ]
    )


def test_ranking_keeps_only_proteins_contributing_final_ligands(tmp_path) -> None:
    known = pd.DataFrame(
        [
            {
                "qseqid": "query-1",
                "sseqid": "P1",
                "search_type": "sequence",
                "uniprot_id": "P1",
                "chem_comp_id": "SHARED",
            },
            {
                "qseqid": "query-1",
                "sseqid": "P2",
                "search_type": "nearest_k",
                "uniprot_id": "P2",
                "chem_comp_id": "SHARED",
            },
        ]
    )
    predicted = pd.DataFrame(
        [
            {
                "qseqid": "query-1",
                "sseqid": "P3",
                "search_type": "domain",
                "uniprot_id": "P3",
                "predicted_chem_comp_id": "PREDICTED-1",
                "tanimoto": 0.9,
            }
        ]
    )

    summary = _process_single_query(
        qseqid="query-1",
        df_cand_q=_candidates(),
        known_q=known,
        predicted_q=predicted,
        known_db_cols=["uniprot_id", "chem_comp_id"],
        known_ligand_col="chem_comp_id",
        predicted_ligand_col="predicted_chem_comp_id",
        search_results_dir=tmp_path,
    )

    ranking = pd.read_csv(tmp_path / "query-1" / "protein_ranking.tsv", sep="\t")
    assert ranking[["protein_rank", "sseqid"]].to_dict("records") == [
        {"protein_rank": 1, "sseqid": "P1"},
        {"protein_rank": 2, "sseqid": "P3"},
    ]
    assert summary["n_proteins_sequence"] == 1
    assert summary["n_proteins_nearest_k"] == 0
    assert summary["n_proteins_domain"] == 1

    known_out = pd.read_csv(tmp_path / "query-1" / "known_ligands.tsv", sep="\t")
    assert known_out["uniprot_id"].tolist() == ["P1"]
    predicted_out = pd.read_csv(
        tmp_path / "query-1" / "predicted_ligands.tsv", sep="\t"
    )
    assert predicted_out["uniprot_id"].tolist() == ["P3"]


def test_ranking_file_is_empty_when_candidates_contribute_no_ligands(tmp_path) -> None:
    summary = _process_single_query(
        qseqid="query-1",
        df_cand_q=_candidates().iloc[[0]].copy(),
        known_q=pd.DataFrame(columns=["search_type", "uniprot_id", "chem_comp_id"]),
        predicted_q=pd.DataFrame(
            columns=["search_type", "uniprot_id", "predicted_chem_comp_id"]
        ),
        known_db_cols=["uniprot_id", "chem_comp_id"],
        known_ligand_col="chem_comp_id",
        predicted_ligand_col="predicted_chem_comp_id",
        search_results_dir=tmp_path,
    )

    ranking_path = tmp_path / "query-1" / "protein_ranking.tsv"
    assert ranking_path.is_file()
    assert pd.read_csv(ranking_path, sep="\t").empty
    assert summary["n_proteins_sequence"] == 0


def test_disabling_ligand_deduplication_keeps_each_contributing_protein(tmp_path) -> None:
    candidates = _candidates().iloc[:2].copy()
    known = pd.DataFrame(
        [
            {
                "search_type": row.search_type,
                "uniprot_id": row.sseqid,
                "chem_comp_id": "SHARED",
            }
            for row in candidates.itertuples()
        ]
    )

    _process_single_query(
        qseqid="query-1",
        df_cand_q=candidates,
        known_q=known,
        predicted_q=pd.DataFrame(
            columns=["search_type", "uniprot_id", "predicted_chem_comp_id"]
        ),
        known_db_cols=["uniprot_id", "chem_comp_id"],
        known_ligand_col="chem_comp_id",
        predicted_ligand_col="predicted_chem_comp_id",
        search_results_dir=tmp_path,
        drop_duplicates=False,
    )

    ranking = pd.read_csv(tmp_path / "query-1" / "protein_ranking.tsv", sep="\t")
    assert ranking["sseqid"].tolist() == ["P1", "P2"]
