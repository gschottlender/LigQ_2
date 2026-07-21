from typing import Any, Optional
from pydantic import BaseModel


class Pagination(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int


class PaginatedResponse(BaseModel):
    data: list[dict[str, Any]]
    pagination: Pagination


class QuerySummary(BaseModel):
    qseqid: str
    n_proteins_sequence: int = 0
    n_proteins_nearest_k: int = 0
    n_proteins_domain: int = 0
    n_known_ligands_sequence: int = 0
    n_known_ligands_nearest_k: int = 0
    n_known_ligands_domain: int = 0
    n_predicted_ligands_sequence: int = 0
    n_predicted_ligands_nearest_k: int = 0
    n_predicted_ligands_domain: int = 0
    has_protein_ranking: bool = False
    has_known_ligands: bool = False
    has_predicted_ligands: bool = False


class SummaryResponse(BaseModel):
    queries: list[QuerySummary]