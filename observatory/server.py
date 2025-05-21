import os
from typing import List, Optional

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatrixRow(BaseModel):
    policy_uri: str
    eval_name: str
    value: float
    replay_url: Optional[str] = None


@app.get("/api/metrics")
async def get_metrics() -> List[str]:
    db_uri = os.getenv("EVAL_DB_URI")
    if not db_uri:
        raise HTTPException(status_code=500, detail="EVAL_DB_URI environment variable not set")

    try:
        conn = duckdb.connect()
        conn.execute(f"ATTACH DATABASE '{db_uri}' AS eval")
        conn.execute("USE eval")

        # Query distinct metrics
        result = conn.execute("""
            SELECT DISTINCT metric 
            FROM episode_metrics 
            ORDER BY metric
        """).fetchall()

        metrics = [row[0] for row in result]
        conn.close()
        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/policy-evals")
async def get_policy_evals(metric: str = "reward") -> List[MatrixRow]:
    db_uri = os.getenv("EVAL_DB_URI")
    if not db_uri:
        raise HTTPException(status_code=500, detail="EVAL_DB_URI environment variable not set")

    try:
        conn = duckdb.connect()
        conn.execute(f"ATTACH DATABASE '{db_uri}' AS eval")
        conn.execute("USE eval")

        # Query data for heatmap with parameterized metric
        result = conn.execute(
            """
          WITH my_metric AS (SELECT * FROM episode_metrics WHERE metric = ?) 
          SELECT
            e.policy_key || ':v' || e.policy_version AS policy_uri, 
            e.eval_name, 
            AVG(m.value) as value, 
            ANY_VALUE(e.replay_url) AS replay_url 
          FROM my_metric m 
          JOIN episode_info e 
          ON m.episode_id = e.episode_id 
          GROUP BY e.eval_name, e.policy_key, e.policy_version
        """,
            [metric],
        ).fetchall()

        # Convert to list of MatrixRow objects
        matrix = [MatrixRow(policy_uri=row[0], eval_name=row[1], value=row[2], replay_url=row[3]) for row in result]

        conn.close()
        return matrix

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
