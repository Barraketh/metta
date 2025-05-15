import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import wandb

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class SimulationMetrics:
    """Container for simulation metrics."""

    steps: int
    elapsed_time: float
    sps: float
    reset_step: int | None = None
    interval_start_time: float = 0.0
    interval_end_time: float = 0.0


def get_next_run_id() -> int:
    """Get the next run ID from the counter file and increment it."""
    counter_file = Path(__file__).parent / ".run_counter"

    # Create counter file if it doesn't exist
    if not counter_file.exists():
        counter_file.write_text("0")

    # Read current counter
    current_id = int(counter_file.read_text().strip())

    # Increment and save
    next_id = current_id + 1
    counter_file.write_text(str(next_id))

    return next_id


def _run_simulation_loop(
    env: MettaGridEnv,
    duration_seconds: float,
    log_interval_seconds: float,
    reset_interval: int | None = None,
    simulation_type: Literal["continuous", "with_resets"] = "continuous",
) -> tuple[int, float, float, list[SimulationMetrics]]:
    """Helper function to run the simulation loop with optional resets.

    Returns:
        tuple[int, float, float, list[SimulationMetrics]]: (total_steps, total_time, average_sps, metrics_history)
    """
    logging.info(
        f"Starting simulation loop. Duration: {duration_seconds}s. Log interval: {log_interval_seconds}s. Reset interval: {reset_interval if reset_interval else 'N/A'}."
    )
    count = 0
    start_time = time.monotonic()
    last_log_time = start_time
    last_log_step_count = 0
    metrics_history: list[SimulationMetrics] = []

    # Track the last reset time to ensure we're not measuring reset overhead
    last_reset_time = start_time

    while time.monotonic() - start_time < duration_seconds:
        current_time = time.monotonic()

        # Check if we need to reset
        if reset_interval and count > 0 and count % reset_interval == 0:
            reset_start = time.monotonic()
            env.reset()
            reset_end = time.monotonic()
            reset_duration = reset_end - reset_start

            logging.info(f"Environment reset at step {count}. Reset took {reset_duration:.3f}s")

            # Only log metrics if we're not in the middle of a measurement interval
            if current_time - last_log_time >= log_interval_seconds:
                metrics_history.append(
                    SimulationMetrics(
                        steps=count,
                        elapsed_time=current_time - start_time,
                        sps=0.0,  # Reset doesn't affect SPS
                        reset_step=count,
                        interval_start_time=reset_start,
                        interval_end_time=reset_end,
                    )
                )

            last_reset_time = reset_end
            last_log_time = reset_end  # Adjust the last log time to account for reset

        # Take a step
        step_start = time.monotonic()
        action = env.action_space.sample()
        env.step(action)
        step_end = time.monotonic()
        count += 1

        # Check if we need to log metrics
        current_time = time.monotonic()
        time_since_last_log = current_time - last_log_time

        if time_since_last_log >= log_interval_seconds:
            steps_this_interval = count - last_log_step_count
            # Only count time since last reset to avoid measuring reset overhead
            effective_time = current_time - max(last_log_time, last_reset_time)
            sps = steps_this_interval / effective_time if effective_time > 0 else 0

            logging.info(
                f"SPS: {sps:.2f} (Total steps: {count}, Elapsed time: {current_time - start_time:.2f}s, "
                f"Steps in interval: {steps_this_interval}, Effective time: {effective_time:.2f}s)"
            )

            metrics_history.append(
                SimulationMetrics(
                    steps=count,
                    elapsed_time=current_time - start_time,
                    sps=sps,
                    interval_start_time=last_log_time,
                    interval_end_time=current_time,
                )
            )

            last_log_time = current_time
            last_log_step_count = count

    total_time = time.monotonic() - start_time
    # Calculate average SPS excluding reset overhead
    total_effective_time = total_time
    if reset_interval:
        # Sum up all reset durations
        reset_durations = [
            m.interval_end_time - m.interval_start_time for m in metrics_history if m.reset_step is not None
        ]
        total_effective_time = total_time - sum(reset_durations)

    avg_sps = count / total_effective_time if total_effective_time > 0 else 0

    logging.info(
        f"Finished simulation loop after {count} steps in {total_time:.2f}s "
        f"(effective time: {total_effective_time:.2f}s, avg SPS: {avg_sps:.2f})"
    )
    return count, total_time, avg_sps, metrics_history


def log_metrics_to_wandb(
    metrics_history: list[SimulationMetrics],
    simulation_type: Literal["continuous", "with_resets"],
) -> None:
    """Log metrics to wandb outside of the measured intervals."""
    for metrics in metrics_history:
        if metrics.reset_step is not None:
            # Log reset event
            wandb.log(
                {
                    f"{simulation_type}/reset": True,
                    f"{simulation_type}/reset_step": metrics.reset_step,
                    f"{simulation_type}/elapsed_time": metrics.elapsed_time,
                    f"{simulation_type}/reset_duration": metrics.interval_end_time - metrics.interval_start_time,
                }
            )
        else:
            # Log SPS metrics
            wandb.log(
                {
                    f"{simulation_type}/sps": metrics.sps,
                    f"{simulation_type}/total_steps": metrics.steps,
                    f"{simulation_type}/elapsed_time": metrics.elapsed_time,
                    f"{simulation_type}/interval_duration": metrics.interval_end_time - metrics.interval_start_time,
                }
            )


def run_continuous_simulation(
    env_config_name: str = "benchmark",
    duration_seconds: float = 60.0,
    log_interval_seconds: float = 10.0,
) -> tuple[int, float, float]:
    logging.info("--- Starting Continuous Simulation ---")
    cfg = get_cfg(env_config_name)
    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    env.reset()
    logging.info("Environment reset for continuous simulation.")
    steps, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env,
        duration_seconds=duration_seconds,
        log_interval_seconds=log_interval_seconds,
        reset_interval=None,
        simulation_type="continuous",
    )
    logging.info(
        f"--- Continuous Simulation Finished --- Total Steps: {steps}, Time Taken: {time_taken:.2f}s, Average SPS: {avg_sps:.2f}"
    )

    # Log metrics to wandb after simulation is complete
    log_metrics_to_wandb(metrics_history, "continuous")

    # Log final metrics
    wandb.log(
        {
            "continuous/final_steps": steps,
            "continuous/final_time": time_taken,
            "continuous/final_avg_sps": avg_sps,
        }
    )
    return steps, time_taken, avg_sps


def run_simulation_with_resets(
    env_config_name: str = "benchmark",
    duration_seconds: float = 60.0,
    log_interval_seconds: float = 10.0,
    reset_every_n_steps: int = 1000,
) -> tuple[int, float, float]:
    logging.info(f"--- Starting Simulation with Resets (every {reset_every_n_steps} steps) ---")
    cfg = get_cfg(env_config_name)
    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    env.reset()
    logging.info("Environment reset for simulation with resets.")
    steps, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env,
        duration_seconds=duration_seconds,
        log_interval_seconds=log_interval_seconds,
        reset_interval=reset_every_n_steps,
        simulation_type="with_resets",
    )
    logging.info(
        f"--- Simulation with Resets Finished --- Total Steps: {steps}, Time Taken: {time_taken:.2f}s, Average SPS: {avg_sps:.2f}"
    )

    # Log metrics to wandb after simulation is complete
    log_metrics_to_wandb(metrics_history, "with_resets")

    # Log final metrics
    wandb.log(
        {
            "with_resets/final_steps": steps,
            "with_resets/final_time": time_taken,
            "with_resets/final_avg_sps": avg_sps,
        }
    )
    return steps, time_taken, avg_sps


def parse_args():
    parser = argparse.ArgumentParser(description="Run MettaGrid benchmark simulations")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of each simulation in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=10.0,
        help="Interval between SPS logging in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=1000,
        help="Number of steps between resets in the reset simulation (default: 1000)",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="benchmark",
        help="Environment configuration name (default: benchmark)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info("Starting benchmark script.")

    # Get next run ID and create run name
    run_id = get_next_run_id()
    run_name = f"dff_metagrid_bench_{run_id:04d}"

    # Initialize wandb
    wandb.init(
        project="metta",
        name=run_name,
        config={
            "duration_seconds": args.duration,
            "log_interval_seconds": args.log_interval,
            "reset_interval": args.reset_interval,
            "env_config": args.env_config,
            "run_id": run_id,
        },
    )

    # Run simulations
    cont_steps, cont_time, cont_sps = run_continuous_simulation(
        env_config_name=args.env_config,
        duration_seconds=args.duration,
        log_interval_seconds=args.log_interval,
    )

    # Add a small delay or clear visual separation in logs if needed
    logging.info("\n" + "=" * 50 + "\n")

    reset_steps, reset_time, reset_sps = run_simulation_with_resets(
        env_config_name=args.env_config,
        duration_seconds=args.duration,
        log_interval_seconds=args.log_interval,
        reset_every_n_steps=args.reset_interval,
    )

    # Log comparison metrics
    wandb.log(
        {
            "comparison/continuous_sps": cont_sps,
            "comparison/reset_sps": reset_sps,
            "comparison/sps_ratio": cont_sps / reset_sps if reset_sps > 0 else float("inf"),
        }
    )

    # Close wandb
    wandb.finish()

    logging.info("Benchmark script finished.")


if __name__ == "__main__":
    main()
