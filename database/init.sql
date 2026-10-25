CREATE TABLE IF NOT EXISTS verification_requests (
    id BIGSERIAL PRIMARY KEY,

    query TEXT NOT NULL,

    answer TEXT NOT NULL,

    verification_status VARCHAR(50),

    confidence DOUBLE PRECISION,

    hallucination_score DOUBLE PRECISION,

    citations JSONB,

    quality_score DOUBLE PRECISION,

    quality_pass BOOLEAN,

    quality_metrics JSONB,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);