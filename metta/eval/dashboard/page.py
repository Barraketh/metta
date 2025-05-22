"""
High‑level report generator.

At the moment we produce a single heat‑map, but the structure anticipates
multiple chart types – simply append more HTML snippets to `graphs_html`.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel

from metta.eval.eval_stats_db import EvalStatsDB, PolicyEvalMetric
from metta.util.config import Config
from mettagrid.util.file import write_data

logger = logging.getLogger(__name__)


class DashboardConfig(Config):
    eval_db_uri: str
    output_path: str = "/tmp/dashboard_data.json"


class DashboardData(BaseModel):
    policy_eval_metrics: List[PolicyEvalMetric]


def generate_dashboard(dashboard_cfg: DashboardConfig):
    with EvalStatsDB.from_uri(dashboard_cfg.eval_db_uri) as db:
        metrics = db.get_avg_metrics_by_policy_and_eval()
        content = DashboardData(policy_eval_metrics=metrics).model_dump_json()

    write_data(dashboard_cfg.output_path, content, content_type="application/json")
