from pydantic import BaseModel


class RepresentationInfo(BaseModel):
    name: str
    metric: str  # "tanimoto" | "cosine"
    default_threshold: float | None = None


class DatabasesResponse(BaseModel):
    databases: list[str]


class RepresentationsResponse(BaseModel):
    representations: list[RepresentationInfo]


class ColumnsResponse(BaseModel):
    columns: list[str]
