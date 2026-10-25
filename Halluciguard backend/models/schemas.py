from pydantic import BaseModel, Field, field_validator


class VerifyRequest(BaseModel):
    query: str = Field(..., min_length=1)

    chat_history: list[dict[str, str]] = Field(
        default_factory=list
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Query cannot be empty or contain only whitespace."
            )

        return value



class VerifyResponse(BaseModel):
    answer: str
    verification_status: str
    confidence: float = Field(ge=0.0, le=1.0)
    hallucination_score: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    citations: list[str] = Field(default_factory=list)
    chat_history: list[dict[str, str]] = Field(
        default_factory=list
    )

    quality_score: float = 0.0
    quality_pass: bool = False
    quality_metrics: dict = Field(default_factory=dict)

    # Additional fields for the verification modal
    initial_response: str | None = None
    final_response: str | None = None
    claims: list[dict] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    correction_summary: dict | None = None
