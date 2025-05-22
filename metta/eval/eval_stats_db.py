"""
EvalStatsDb adds views on top of SimulationStatsDb
to make it easier to query policy performance across simulations,
while handling the fact that some metrics are only logged when non‑zero.

Normalisation rule
------------------
For every query we:
1.  Count the **potential** agent‑episode samples for the policy / filter.
2.  Aggregate the recorded metric values (missing = 0).
3.  Divide by the potential count.

This yields a true mean even when zeros are omitted from logging.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

from pydantic import BaseModel

from metta.agent.policy_store import PolicyRecord
from metta.sim.simulation_stats_db import SimulationStatsDB
from mettagrid.util.file import local_copy


class PolicyEvalMetric(BaseModel):
    policy_uri: str
    eval_name: str
    metric: str
    value: float
    replay_url: str | None


class EvalStatsDB:
    # ------------------------------------------------------------------ #
    #   Construction / schema                                            #
    # ------------------------------------------------------------------ #
    def __init__(self, sim_stats_db: SimulationStatsDB) -> None:
        self.sim_stats_db = sim_stats_db

        # Create views
        self.sim_stats_db.con.execute(
            """
            CREATE VIEW IF NOT EXISTS episode_info AS (
                WITH episode_agents AS (
                  SELECT episode_id,
                  COUNT(*) as num_agents 
                  FROM agent_policies 
                  GROUP BY episode_id
                )
                SELECT 
                  e.id as episode_id,
                  s.name as eval_name,
                  s.suite,
                  s.env,
                  s.policy_key,
                  s.policy_version,
                  e.created_at,
                  e.replay_url,
                  episode_agents.num_agents 
                FROM simulations s 
                JOIN episodes e ON e.simulation_id = s.id 
                JOIN episode_agents ON e.id = episode_agents.episode_id)
            """
        )

        self.sim_stats_db.con.execute(
            """
            CREATE VIEW IF NOT EXISTS episode_metrics AS (
                WITH totals AS (
                    SELECT episode_id, metric, SUM(value) as value 
                    FROM agent_metrics 
                    GROUP BY episode_id, metric
                ) 
                SELECT t.episode_id, t.metric, t.value / e.num_agents as value 
                FROM totals t 
                JOIN episode_info e ON t.episode_id = e.episode_id)
            """
        )

    @classmethod
    @contextmanager
    def from_uri(cls, path: str):
        """Download (if remote), open, and yield an EvalStatsDB."""
        with local_copy(path) as local_path:
            sim_stats_db = SimulationStatsDB(local_path)
            yield cls(sim_stats_db)

    # Public alias (referenced by downstream code/tests)
    def potential_samples_for_metric(
        self,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> int:
        pass

    def count_metric_agents(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        filter_condition: str | None = None,
    ) -> int:
        pass

    # Convenience wrappers ------------------------------------------------
    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pass

    def get_std_metric_by_filter(
        self,
        metric: str,
        policy_record: PolicyRecord,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pass

    # ------------------------------------------------------------------ #
    #   Utilities                                                        #
    # ------------------------------------------------------------------ #
    def sample_count(
        self,
        policy_record: Optional[PolicyRecord] = None,
        sim_suite: Optional[str] = None,
        sim_name: Optional[str] = None,
        sim_env: Optional[str] = None,
    ) -> int:
        pass

    # ------------------------------------------------------------------ #
    #   Per‑simulation breakdown                                         #
    # ------------------------------------------------------------------ #
    def simulation_scores(self, policy_record: PolicyRecord, metric: str) -> Dict[tuple, float]:
        pass

    def get_avg_metrics_by_policy_and_eval(self) -> List[PolicyEvalMetric]:
        query = """
            SELECT
              e.policy_key || ':v' || e.policy_version AS policy_uri, 
              e.eval_name,
              m.metric,
              AVG(m.value) as value, 
              ANY_VALUE(e.replay_url) AS replay_url
            FROM episode_metrics m 
            JOIN episode_info e 
            ON m.episode_id = e.episode_id 
            GROUP BY e.eval_name, e.policy_key, e.policy_version, m.metric
        """
        rows = self.sim_stats_db.con.execute(query).fetchall()
        return [
            PolicyEvalMetric(policy_uri=row[0], eval_name=row[1], metric=row[2], value=row[3], replay_url=row[4])
            for row in rows
        ]
