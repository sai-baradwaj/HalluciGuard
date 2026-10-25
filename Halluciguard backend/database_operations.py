from database import get_connection
import json


def save_verification(query: str, result: dict):
    conn = get_connection()

    try:
        cur = conn.cursor()

        sql = """
            INSERT INTO verification_requests (
                query,
                answer,
                verification_status,
                confidence,
                hallucination_score,
                citations,
                quality_score,
                quality_pass,
                quality_metrics
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id;
        """

        cur.execute(
            sql,
            (
                query,
                result.get("answer", ""),
                result.get("verification_status"),
                result.get("confidence"),
                result.get("hallucination_score"),
                json.dumps(result.get("citations", [])),
                result.get("quality_score"),
                result.get("quality_pass"),
                json.dumps(result.get("quality_metrics", {})),
            )
        )

        record_id = cur.fetchone()[0]

        conn.commit()

        cur.close()

        return record_id

    finally:
        conn.close()