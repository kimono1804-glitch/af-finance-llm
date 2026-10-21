"""Pydantic schemas shared across data generation, the calculator tool, ingestion and inference."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

class CalcStep(BaseModel):
    """One calculation. Inputs are numbers, '@step_id' references, or lists of those.

    Keys ending in `_pct` take a percentage as written in the source (3.5 means 3.5%).
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    formula: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class CalcRequest(BaseModel):
    steps: list[CalcStep] = Field(min_length=1, max_length=25)


class StepResult(BaseModel):
    formula: str
    value: float
    display: str


class CalcResponse(BaseModel):
    results: dict[str, StepResult] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# Table chunks (RAG + training contexts use the same shape)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    source: str
    sheet: str
    section: str
    periods: list[str] = Field(default_factory=list)
    row_labels: list[str] = Field(default_factory=list)
    units: str = ""
    page: int | None = None


class TableChunk(BaseModel):
    chunk_id: str
    title: str
    text: str
    metadata: ChunkMetadata
