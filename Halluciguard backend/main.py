from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import API_KEY
from models.schemas import VerifyRequest, VerifyResponse
from workflow.graph import run_halluciguard

from cache import (
    get_hallucination_cache,
    set_hallucination_cache,
)

from database_operations import save_verification

from logger import logger


app = FastAPI(
    title="HalluciGuard Backend API",
    description="Backend API for the HalluciGuard hallucination detection system",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "HalluciGuard Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "halluciguard-backend",
    }


def authenticate(api_key: str | None) -> None:
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


@app.post(
    "/verify",
    response_model=VerifyResponse,
)
def verify_query(
    request: VerifyRequest,
    x_api_key: str | None = Header(default=None),
):
    # ---------------------------------------------------------
    # 1. Authenticate request
    # ---------------------------------------------------------

    authenticate(x_api_key)

    # ---------------------------------------------------------
    # 2. Check Redis cache
    # ---------------------------------------------------------
    # BUG FIX: this lookup was entirely commented out, so caching was
    # write-only (every request re-ran the full LangGraph workflow even
    # for a repeated query). Restored the read path. The cached value is
    # validated through VerifyResponse before being returned so a stale
    # or malformed cache entry can never be served as-is; any validation
    # failure falls through to a fresh workflow run instead of crashing.

    try:
        cached_result = get_hallucination_cache(request.query)

        if cached_result is not None:
            try:
                validated_cached = VerifyResponse.model_validate(cached_result)
            except Exception as validation_error:
                logger.error(
                    f"Discarding malformed cache entry for query "
                    f"'{request.query}': {validation_error}"
                )
            else:
                logger.info(
                    f"Returning cached result for query: {request.query}"
                )
                return validated_cached

    except Exception as cache_error:
        logger.error(
            f"Redis cache lookup failed: {cache_error}"
        )

        # Cache failure should NOT stop HalluciGuard.
        # We continue to the actual verification workflow.



    # ---------------------------------------------------------
    # 3. Run HalluciGuard LangGraph workflow
    # ---------------------------------------------------------

    try:
        logger.info(
            f"Running HalluciGuard workflow for query: {request.query}"
        )

        result = run_halluciguard(
            request.query,
            chat_history=request.chat_history,
        )
        validated_result = VerifyResponse.model_validate(result)

    except HTTPException:
        raise

    except Exception as workflow_error:
        logger.exception(
            "HalluciGuard workflow failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to process the query: {type(workflow_error).__name__}: {workflow_error}",
        )



    # ---------------------------------------------------------
    # 4. Save verification result to PostgreSQL
    # ---------------------------------------------------------

    try:
        record_id = save_verification(
            request.query,
            validated_result.model_dump(),
        )

        logger.info(
            f"Verification result saved to PostgreSQL. ID: {record_id}"
        )

    except Exception as database_error:
        logger.error(
            f"PostgreSQL save failed: {database_error}"
        )

        # Database failure should NOT destroy an otherwise
        # successful verification response.


    # ---------------------------------------------------------
    # 5. Store result in Redis cache
    # ---------------------------------------------------------

    try:
        set_hallucination_cache(
            request.query,
            validated_result.model_dump(),
        )

        logger.info(
            f"Verification result cached for query: {request.query}"
        )

    except Exception as cache_error:
        logger.error(
            f"Redis cache save failed: {cache_error}"
        )

        # Cache failure should NOT destroy the response.


    # ---------------------------------------------------------
    # 6. Return final verification result
    # ---------------------------------------------------------

    return validated_result