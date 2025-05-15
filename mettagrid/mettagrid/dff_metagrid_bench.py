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
    """Container for simulation metrics after simplification."""

    steps_in_interval: int  # Number of steps taken in this specific logging interval
    time_for_interval: float  # Time taken for this specific logging interval (seconds)
    sps_this_interval: float  # SPS calculated for this interval
    total_steps_so_far: int  # Cumulative steps from the start of the simulation
    total_elapsed_time: float  # Cumulative time from the start of the simulation
    reset_time_this_cycle: float | None = None  # Time taken for reset, if one occurred in this interval
    reset_occurred_at_step: int | None = None  # Step number at which reset occurred


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
    total_simulation_steps: int,
    log_interval_steps: int,
    steps_per_reset_cycle: int | None = None,
    simulation_type: Literal["continuous", "with_resets"] = "continuous",
) -> tuple[int, float, float, list[SimulationMetrics]]:
    """Helper function to run the simulation loop based on total steps.

    Returns:
        tuple[int, float, float, list[SimulationMetrics]]:
            (total_steps_executed, total_time_seconds, overall_average_sps, metrics_history)
    """
    logging.info(
        f"Starting simulation loop. Total steps: {total_simulation_steps}. "
        f"Log interval: {log_interval_steps} steps. "
        f"Steps per reset cycle: {steps_per_reset_cycle if steps_per_reset_cycle else 'N/A'}."
    )

    current_total_steps = 0
    simulation_start_time = time.monotonic()

    interval_start_time = simulation_start_time
    steps_at_interval_start = 0

    metrics_history: list[SimulationMetrics] = []
    total_reset_time = 0.0

    while current_total_steps < total_simulation_steps:
        steps_to_run_this_cycle = log_interval_steps
        actual_reset_time_this_interval: float | None = None
        actual_reset_step_this_interval: int | None = None

        # Determine if a reset should happen within this logging interval
        if simulation_type == "with_resets" and steps_per_reset_cycle:
            steps_into_current_reset_cycle = current_total_steps % steps_per_reset_cycle
            if (
                steps_into_current_reset_cycle == 0 and current_total_steps > 0
            ):  # Reset at start of cycle, not at step 0
                reset_start_time = time.monotonic()
                env.reset()
                reset_end_time = time.monotonic()
                actual_reset_time_this_interval = reset_end_time - reset_start_time
                total_reset_time += actual_reset_time_this_interval
                actual_reset_step_this_interval = current_total_steps
                logging.info(
                    f"Environment reset at step {current_total_steps}. Reset took {actual_reset_time_this_interval:.3f}s"
                )
                # Adjust interval_start_time to not include reset time in SPS calculation for this step
                interval_start_time = reset_end_time

        # Run steps for the current logging interval or until total_simulation_steps is reached
        steps_remaining_in_simulation = total_simulation_steps - current_total_steps
        steps_this_interval_actual = 0

        # Cap steps_to_run_this_cycle by remaining simulation steps
        steps_to_execute_now = min(steps_to_run_this_cycle, steps_remaining_in_simulation)

        for _ in range(steps_to_execute_now):
            if current_total_steps >= total_simulation_steps:
                break  # Ensure we don't exceed total_simulation_steps
            action = env.action_space.sample()
            env.step(action)
            current_total_steps += 1
            steps_this_interval_actual += 1

        interval_end_time = time.monotonic()
        interval_duration = interval_end_time - interval_start_time

        # Avoid division by zero if interval_duration is too small (e.g., due to reset)
        sps_this_interval = steps_this_interval_actual / interval_duration if interval_duration > 1e-9 else 0.0

        current_simulation_elapsed_time = interval_end_time - simulation_start_time

        logging.info(
            f"Interval SPS: {sps_this_interval:.2f} "
            f"(Steps in interval: {steps_this_interval_actual}, Interval time: {interval_duration:.2f}s, "
            f"Total steps: {current_total_steps}, Total time: {current_simulation_elapsed_time:.2f}s)"
        )

        metrics_history.append(
            SimulationMetrics(
                steps_in_interval=steps_this_interval_actual,
                time_for_interval=interval_duration,
                sps_this_interval=sps_this_interval,
                total_steps_so_far=current_total_steps,
                total_elapsed_time=current_simulation_elapsed_time,
                reset_time_this_cycle=actual_reset_time_this_interval,
                reset_occurred_at_step=actual_reset_step_this_interval,
            )
        )

        interval_start_time = interval_end_time  # Start of next interval is end of current
        steps_at_interval_start = current_total_steps

        # If a reset happened, ensure the next interval starts its time measurement *after* the reset
        if actual_reset_time_this_interval is not None:
            interval_start_time = time.monotonic()  # Re-align start time after a reset

    total_simulation_time_seconds = time.monotonic() - simulation_start_time
    # Effective time for SPS calculation excludes time spent on resets
    effective_simulation_time = total_simulation_time_seconds - total_reset_time
    overall_avg_sps = current_total_steps / effective_simulation_time if effective_simulation_time > 0 else 0

    logging.info(
        f"Finished simulation loop. Executed {current_total_steps} steps in {total_simulation_time_seconds:.2f}s. "
        f"Total reset time: {total_reset_time:.3f}s. "
        f"Effective simulation time (excluding resets): {effective_simulation_time:.2f}s. "
        f"Overall Average SPS (excluding resets): {overall_avg_sps:.2f}"
    )
    return current_total_steps, total_simulation_time_seconds, overall_avg_sps, metrics_history


def log_metrics_to_wandb(
    metrics_history: list[SimulationMetrics],
    simulation_type: Literal["continuous", "with_resets"],
) -> None:
    """Log metrics to wandb based on the new SimulationMetrics structure."""
    for metrics in metrics_history:
        log_data = {
            f"{simulation_type}/interval_sps": metrics.sps_this_interval,
            f"{simulation_type}/steps_in_interval": metrics.steps_in_interval,
            f"{simulation_type}/time_for_interval": metrics.time_for_interval,
            f"{simulation_type}/total_steps_so_far": metrics.total_steps_so_far,
            f"{simulation_type}/total_elapsed_time": metrics.total_elapsed_time,
        }
        if metrics.reset_time_this_cycle is not None:
            log_data[f"{simulation_type}/reset_duration"] = metrics.reset_time_this_cycle
            log_data[f"{simulation_type}/reset_occurred_at_step"] = metrics.reset_occurred_at_step

        wandb.log(log_data)


def run_continuous_simulation(
    env_config_name: str = "benchmark",
    total_steps: int = 10000,  # Default to 10k steps
    log_interval_steps: int = 1000,  # Default to log every 1k steps
) -> tuple[int, float, float]:
    logging.info("--- Starting Continuous Simulation ---")
    cfg = get_cfg(env_config_name)
    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    env.reset()  # Initial reset
    logging.info(
        f"Environment reset for continuous simulation. Target steps: {total_steps}, Log interval: {log_interval_steps} steps."
    )

    steps_executed, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env,
        total_simulation_steps=total_steps,
        log_interval_steps=log_interval_steps,
        simulation_type="continuous",
    )
    logging.info(
        f"--- Continuous Simulation Finished --- Executed Steps: {steps_executed}, Time Taken: {time_taken:.2f}s, Average SPS: {avg_sps:.2f}"
    )

    log_metrics_to_wandb(metrics_history, "continuous")

    wandb.log(
        {
            "continuous/final_total_steps": steps_executed,
            "continuous/final_total_time": time_taken,
            "continuous/final_avg_sps": avg_sps,
        }
    )
    return steps_executed, time_taken, avg_sps


def run_simulation_with_resets(
    env_config_name: str = "benchmark",
    total_steps: int = 100000,  # Default to 100k total steps
    log_interval_steps: int = 1000,  # Default to log every 1k steps
    steps_per_reset_cycle: int = 1000,  # Default to reset every 1k steps
) -> tuple[int, float, float]:
    logging.info(f"--- Starting Simulation with Resets (every {steps_per_reset_cycle} steps) ---")
    cfg = get_cfg(env_config_name)
    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    env.reset()  # Initial reset
    logging.info(
        f"Environment reset for simulation with resets. Target steps: {total_steps}, "
        f"Log interval: {log_interval_steps} steps, Reset every: {steps_per_reset_cycle} steps."
    )

    steps_executed, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env,
        total_simulation_steps=total_steps,
        log_interval_steps=log_interval_steps,
        steps_per_reset_cycle=steps_per_reset_cycle,
        simulation_type="with_resets",
    )
    logging.info(
        f"--- Simulation with Resets Finished --- Executed Steps: {steps_executed}, Time Taken: {time_taken:.2f}s, Average SPS: {avg_sps:.2f}"
    )

    log_metrics_to_wandb(metrics_history, "with_resets")

    wandb.log(
        {
            "with_resets/final_total_steps": steps_executed,
            "with_resets/final_total_time": time_taken,
            "with_resets/final_avg_sps": avg_sps,
        }
    )
    return steps_executed, time_taken, avg_sps


def parse_args():
    parser = argparse.ArgumentParser(description="Run MettaGrid benchmark simulations")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total number of simulation steps to run (default: 10000)",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=1000,
        help="Interval for logging SPS in steps (default: 1000)",
    )
    parser.add_argument(
        "--steps-per-reset-cycle",
        type=int,
        default=1000,
        help="Number of steps between resets in the 'with_resets' simulation (default: 1000)",
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
    logging.info(
        "Starting benchmark script with step-based configuration. Both continuous and with-resets simulations will be run in a single wandb run."
    )

    run_id = get_next_run_id()
    run_name = f"dff_metagrid_bench_steps_{run_id:04d}_combined"

    # Initialize wandb once for both simulations
    wandb.init(
        project="metta",
        name=run_name,
        config={
            "total_steps": args.total_steps,
            "log_interval_steps": args.log_interval_steps,
            "steps_per_reset_cycle": args.steps_per_reset_cycle,  # Relevant for the with_resets part
            "env_config": args.env_config,
            "run_id": run_id,  # Add run_id to config for easier tracking
        },
    )

    logging.info(f"Initiating combined benchmark with wandb run name: {run_name}")

    # Run continuous simulation
    logging.info("--- Running Continuous Simulation Part ---")
    run_continuous_simulation(
        env_config_name=args.env_config,
        total_steps=args.total_steps,
        log_interval_steps=args.log_interval_steps,
    )
    # Separator in logs for clarity if needed, wandb metrics are already namespaced
    logging.info("--- Continuous Simulation Part Finished ---")

    # Run simulation with resets
    logging.info("--- Running Simulation with Resets Part ---")
    run_simulation_with_resets(
        env_config_name=args.env_config,
        total_steps=args.total_steps,
        log_interval_steps=args.log_interval_steps,
        steps_per_reset_cycle=args.steps_per_reset_cycle,
    )
    logging.info("--- Simulation with Resets Part Finished ---")

    # Finish the single wandb run
    wandb.finish()

    logging.info("Benchmark script finished. All metrics logged to a single wandb run.")


if __name__ == "__main__":
    main()
