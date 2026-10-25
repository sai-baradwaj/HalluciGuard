from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import API_KEY
from models.schemas import VerifyRequest, VerifyResponse
from workflow.graph import run_halluciguard

from cache import get_hallucination_cache, set_hallucination_cache
from database_operations import save_verification


app = FastAPI(
    title="HalluciGuard Backend API",
    description="Backend API for the HalluciGuard hallucination detection system",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development configuration
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health / Home
# --------------------------------------------------

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


# --------------------------------------------------
# Authentication
# --------------------------------------------------

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


# --------------------------------------------------
# Verification Endpoint
# --------------------------------------------------
@app.post(
    "/verify",
    response_model=VerifyResponse,
)
def verify_query(
    request: VerifyRequest,
    x_api_key: str | None = Header(default=None),
):
    authenticate(x_api_key)

    try:
        query = request.query.strip()

        # -----------------------------------------
        # 1. Check Redis cache
        # -----------------------------------------

        cached_result = get_hallucination_cache(query)

        if cached_result is not None:
            return VerifyResponse.model_validate(cached_result)

        # -----------------------------------------
        # 2. Run HalluciGuard workflow
        # -----------------------------------------

        result = run_halluciguard(query)

        validated_result = VerifyResponse.model_validate(result)

        result_dict = validated_result.model_dump()

        # -----------------------------------------
        # 3. Save result to PostgreSQL
        # -----------------------------------------

        save_verification(
            query=query,
            result=result_dict,
        )

        # -----------------------------------------
        # 4. Save result to Redis
        # -----------------------------------------

        set_hallucination_cache(
            query,
            result_dict,
        )

        # -----------------------------------------
        # 5. Return result
        # -----------------------------------------

        return validated_result

    except HTTPException:
        raise

    except Exception as e:
        print(f"Verification error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query",
        )