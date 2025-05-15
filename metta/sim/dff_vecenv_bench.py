import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pufferlib.vector
import wandb
from omegaconf import DictConfig, OmegaConf

from metta.sim.vecenv import make_vecenv  # Key import for vectorized envs
from mettagrid.config.utils import get_cfg  # Still useful for loading the base env config

# from mettagrid.mettagrid_env import MettaGridEnv # No longer directly using MettaGridEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class SimulationMetrics:
    """Container for simulation metrics after simplification."""

    steps_in_interval: int
    time_for_interval: float
    sps_this_interval: float
    total_steps_so_far: int
    total_elapsed_time: float
    reset_time_this_cycle: float | None = None
    reset_occurred_at_step: int | None = None


def get_next_run_id() -> int:
    """Get the next run ID from the counter file and increment it."""
    counter_file = Path(__file__).parent / ".run_counter_vecenv"  # Separate counter for vecenv benchmarks
    if not counter_file.exists():
        counter_file.write_text("0")
    current_id = int(counter_file.read_text().strip())
    next_id = current_id + 1
    counter_file.write_text(str(next_id))
    return next_id


def _run_simulation_loop(
    env: pufferlib.PufferEnv,  # Env type updated to PufferEnv
    num_envs: int,  # Added to track number of parallel environments
    total_simulation_steps: int,  # Total individual agent steps
    log_interval_steps: int,  # Log based on individual agent steps
    steps_per_reset_cycle: int | None = None,  # Reset cycle based on individual agent steps
    simulation_type: Literal["continuous", "with_resets"] = "continuous",
) -> tuple[int, float, float, list[SimulationMetrics]]:
    """Helper function to run the simulation loop for vectorized environments."""
    logging.info(
        f"Starting vecenv simulation loop. Total individual steps: {total_simulation_steps}. "
        f"Log interval (individual steps): {log_interval_steps}. "
        f"Steps per reset cycle (individual): {steps_per_reset_cycle if steps_per_reset_cycle else 'N/A'}. "
        f"Number of envs: {num_envs}."
    )

    current_total_individual_steps = 0
    simulation_start_time = time.monotonic()
    interval_start_time = simulation_start_time
    metrics_history: list[SimulationMetrics] = []
    total_reset_time = 0.0

    # Initial observations, dones, infos from reset
    # For vecenv, reset() doesn't take arguments typically needed for first call like seed
    obs, info = env.reset()  # puffer reset gives obs, info

    while current_total_individual_steps < total_simulation_steps:
        actual_reset_time_this_interval: float | None = None
        actual_reset_step_this_interval: int | None = None

        # Determine if a reset should happen
        if simulation_type == "with_resets" and steps_per_reset_cycle:
            # Check if any environment *would* cross a reset boundary in this iteration
            # This logic might need refinement for vecenvs; for now, assume a global step counter for resets
            if (
                current_total_individual_steps > 0
                and (current_total_individual_steps // num_envs)
                % (steps_per_reset_cycle // num_envs if steps_per_reset_cycle > num_envs else 1)
                == 0
            ):
                # Approximate reset trigger based on global steps / num_envs
                # This is a simplification. True per-env reset cycles are more complex.
                # For now, we reset all envs together.
                logging.info(
                    f"Global reset triggered for all vectorized environments at individual step ~{current_total_individual_steps}."
                )
                reset_start_time = time.monotonic()
                obs, info = env.reset()  # Reset all envs
                reset_end_time = time.monotonic()
                actual_reset_time_this_interval = reset_end_time - reset_start_time
                total_reset_time += actual_reset_time_this_interval
                actual_reset_step_this_interval = current_total_individual_steps  # Log at the current total step count
                logging.info(
                    f"All environments reset at ~step {current_total_individual_steps}. Reset took {actual_reset_time_this_interval:.3f}s"
                )
                interval_start_time = reset_end_time  # Adjust for time spent in reset

        # Calculate how many *vectorized steps* to run in this logging interval
        # Each vectorized step accounts for `num_envs` individual steps.
        remaining_individual_steps = total_simulation_steps - current_total_individual_steps

        # Number of individual steps intended for this interval before next log
        individual_steps_this_interval_target = log_interval_steps

        # Ensure we don't overshoot total_simulation_steps
        individual_steps_to_execute_now = min(individual_steps_this_interval_target, remaining_individual_steps)

        vec_steps_to_execute_now = (individual_steps_to_execute_now + num_envs - 1) // num_envs  # ceil division

        individual_steps_taken_this_interval = 0
        if vec_steps_to_execute_now <= 0:  # No steps to run if remaining is less than num_envs for a full vec_step
            if remaining_individual_steps > 0:  # if some steps remain but not enough for full vec step
                # This case needs careful handling, could run 1 vec_step and only count actuals
                # For simplicity now, we might over/undershoot slightly if not perfectly divisible
                # Or we simply break if no full vec_step can be made.
                # Let's assume for now we only do full vec_steps within the loop.
                # The final count handles the exact total.
                pass  # Will be handled by the while condition

        loop_start_time_interval = time.monotonic()
        for i in range(vec_steps_to_execute_now):
            if current_total_individual_steps >= total_simulation_steps:
                break

            actions = env.action_space.sample()
            # PufferLib step returns: next_obs, reward, terminated, truncated, info
            next_obs, reward, terminated, truncated, info = env.step(actions)

            # In PufferLib, `terminated` and `truncated` are arrays.
            # `done = terminated | truncated`
            # `env.reset()` is called internally by PufferLib on done.
            # The `info` dict contains `{'elapsed_steps': count}` per sub-environment which can be useful.
            # For SPS, we count each sub-environment's step.
            current_total_individual_steps += num_envs  # Each vec_step processes num_envs individual steps
            individual_steps_taken_this_interval += num_envs
            obs = next_obs  # For the next iteration

            # If any env is done, PufferLib handles reset and gives new obs.
            # No explicit env.reset(idx) call needed here typically for PufferLib.

        interval_end_time = time.monotonic()
        # Adjust for actual steps if loop broke early
        if individual_steps_taken_this_interval > individual_steps_to_execute_now:
            # This can happen if vec_steps_to_execute_now * num_envs > individual_steps_to_execute_now
            # We should use the minimum of what was planned vs what was executed due to num_envs multiple
            individual_steps_taken_this_interval = min(
                individual_steps_taken_this_interval,
                individual_steps_to_execute_now + (num_envs - individual_steps_to_execute_now % num_envs) % num_envs,
            )

        # Duration for SPS calculation should be time spent stepping, excluding explicit resets
        # If a reset happened this interval, interval_start_time was already advanced.
        # loop_duration = interval_end_time - loop_start_time_interval
        # Using interval_start_time which is reset after a reset event.
        interval_duration = interval_end_time - interval_start_time

        sps_this_interval = (
            individual_steps_taken_this_interval / interval_duration if interval_duration > 1e-9 else 0.0
        )
        current_simulation_elapsed_time = interval_end_time - simulation_start_time

        if individual_steps_taken_this_interval > 0:  # Only log if steps were taken
            logging.info(
                f"Interval Individual SPS: {sps_this_interval:.2f} "
                f"(Indiv Steps in interval: {individual_steps_taken_this_interval}, Interval time: {interval_duration:.2f}s, "
                f"Total indiv steps: {current_total_individual_steps}, Total time: {current_simulation_elapsed_time:.2f}s)"
            )
            metrics_history.append(
                SimulationMetrics(
                    steps_in_interval=individual_steps_taken_this_interval,
                    time_for_interval=interval_duration,
                    sps_this_interval=sps_this_interval,
                    total_steps_so_far=current_total_individual_steps,
                    total_elapsed_time=current_simulation_elapsed_time,
                    reset_time_this_cycle=actual_reset_time_this_interval,
                    reset_occurred_at_step=actual_reset_step_this_interval,
                )
            )

        interval_start_time = interval_end_time  # Start of next interval

    total_simulation_time_seconds = time.monotonic() - simulation_start_time
    effective_simulation_time = total_simulation_time_seconds - total_reset_time
    # Ensure current_total_individual_steps accurately reflects only steps up to total_simulation_steps
    final_steps_executed = min(current_total_individual_steps, total_simulation_steps)

    overall_avg_sps = final_steps_executed / effective_simulation_time if effective_simulation_time > 0 else 0.0

    logging.info(
        f"Finished vecenv simulation loop. Executed ~{final_steps_executed} individual steps in {total_simulation_time_seconds:.2f}s. "
        f"Total reset time: {total_reset_time:.3f}s. "
        f"Effective simulation time (excluding resets): {effective_simulation_time:.2f}s. "
        f"Overall Average Individual SPS (excluding resets): {overall_avg_sps:.2f}"
    )
    # Ensure final steps don't exceed requested total simulation steps due to batching
    return final_steps_executed, total_simulation_time_seconds, overall_avg_sps, metrics_history


def log_metrics_to_wandb(
    metrics_history: list[SimulationMetrics],
    simulation_type: Literal["continuous", "with_resets"],
) -> None:
    """Log metrics to wandb."""
    for metrics in metrics_history:
        log_data = {
            f"{simulation_type}/vec_interval_sps": metrics.sps_this_interval,
            f"{simulation_type}/vec_steps_in_interval": metrics.steps_in_interval,
            f"{simulation_type}/vec_time_for_interval": metrics.time_for_interval,
            f"{simulation_type}/vec_total_steps_so_far": metrics.total_steps_so_far,
            f"{simulation_type}/vec_total_elapsed_time": metrics.total_elapsed_time,
        }
        if metrics.reset_time_this_cycle is not None:
            log_data[f"{simulation_type}/vec_reset_duration"] = metrics.reset_time_this_cycle
            log_data[f"{simulation_type}/vec_reset_occurred_at_step"] = metrics.reset_occurred_at_step
        wandb.log(log_data)


def run_continuous_simulation(
    env_config: DictConfig,  # Changed from name to actual DictConfig
    vectorization: str,
    num_envs: int,
    num_workers: int,
    batch_size: int | None,
    total_steps: int,  # Total individual steps
    log_interval_steps: int,  # Log interval for individual steps
) -> tuple[int, float, float]:
    logging.info(
        f"--- Starting Continuous VecEnv Simulation ({vectorization}, {num_envs} envs, {num_workers} workers) ---"
    )

    vec_env = make_vecenv(
        env_cfg=env_config,
        vectorization=vectorization,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,  # Default batch_size to num_envs
        render_mode=None,  # No rendering for benchmark
    )
    # Initial reset is handled by make_vecenv or by the first call in _run_simulation_loop
    logging.info(
        f"Vectorized environment created. Target individual steps: {total_steps}, Log interval (individual steps): {log_interval_steps}."
    )

    steps_executed, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env=vec_env,
        num_envs=num_envs,
        total_simulation_steps=total_steps,
        log_interval_steps=log_interval_steps,
        simulation_type="continuous",
    )
    logging.info(
        f"--- Continuous VecEnv Simulation Finished --- Executed Indiv Steps: {steps_executed}, Time: {time_taken:.2f}s, Avg Indiv SPS: {avg_sps:.2f}"
    )
    log_metrics_to_wandb(metrics_history, "continuous")
    wandb.log(
        {
            "continuous/vec_final_total_steps": steps_executed,
            "continuous/vec_final_total_time": time_taken,
            "continuous/vec_final_avg_sps": avg_sps,
        }
    )
    vec_env.close()
    return steps_executed, time_taken, avg_sps


def run_simulation_with_resets(
    env_config: DictConfig,  # Changed from name to actual DictConfig
    vectorization: str,
    num_envs: int,
    num_workers: int,
    batch_size: int | None,
    total_steps: int,  # Total individual steps
    log_interval_steps: int,  # Log interval for individual steps
    steps_per_reset_cycle: int,  # Reset cycle for individual steps
) -> tuple[int, float, float]:
    logging.info(
        f"--- Starting VecEnv Simulation with Resets ({vectorization}, {num_envs} envs, {num_workers} workers, reset every ~{steps_per_reset_cycle} indiv steps) ---"
    )

    vec_env = make_vecenv(
        env_cfg=env_config,
        vectorization=vectorization,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        render_mode=None,
    )
    logging.info(
        f"Vectorized environment created for simulation with resets. Target indiv steps: {total_steps}, Log interval (indiv): {log_interval_steps}, Reset every (indiv): {steps_per_reset_cycle}."
    )

    steps_executed, time_taken, avg_sps, metrics_history = _run_simulation_loop(
        env=vec_env,
        num_envs=num_envs,
        total_simulation_steps=total_steps,
        log_interval_steps=log_interval_steps,
        steps_per_reset_cycle=steps_per_reset_cycle,
        simulation_type="with_resets",
    )
    logging.info(
        f"--- VecEnv Simulation with Resets Finished --- Executed Indiv Steps: {steps_executed}, Time: {time_taken:.2f}s, Avg Indiv SPS: {avg_sps:.2f}"
    )
    log_metrics_to_wandb(metrics_history, "with_resets")
    wandb.log(
        {
            "with_resets/vec_final_total_steps": steps_executed,
            "with_resets/vec_final_total_time": time_taken,
            "with_resets/vec_final_avg_sps": avg_sps,
        }
    )
    vec_env.close()
    return steps_executed, time_taken, avg_sps


def parse_args():
    parser = argparse.ArgumentParser(description="Run PufferLib Vectorized Environment benchmark simulations")
    parser.add_argument(
        "--env-config-name",
        type=str,
        default="benchmark",
        help="Base environment configuration name (from mettagrid configs, e.g., benchmark)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100000,
        help="Total number of *individual agent* simulation steps (default: 100k)",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=10000,
        help="Interval for logging SPS in *individual agent* steps (default: 10k)",
    )
    parser.add_argument(
        "--steps-per-reset-cycle",
        type=int,
        default=10000,
        help="Number of *individual agent* steps between resets in 'with_resets' simulation (default: 10k)",
    )

    parser.add_argument(
        "--vectorization",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing", "ray"],
        help="Vectorization backend (default: serial)",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments (default: 1)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for multiprocessing/ray (ignored for serial) (default: 1)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for vectorization (default: num_envs)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(
        "Starting PufferLib VecEnv benchmark script. Both continuous and with-resets simulations will be run if configured."
    )

    run_id = get_next_run_id()
    run_name = f"dff_vecenv_bench_{args.vectorization}_ne{args.num_envs}_nw{args.num_workers}_{run_id:04d}"

    # Load the base environment config once
    # Assuming get_cfg loads a config compatible with what make_env_func in vecenv.py expects
    # This might need adjustment if make_env_func expects a hydra-style config object directly
    # For now, we load it as a DictConfig as vecenv.py uses hydra.utils.instantiate
    base_env_cfg_obj: DictConfig = get_cfg(args.env_config_name)  # type: ignore

    # Convert OmegaConf to a serializable dict for wandb logging.
    # The full DictConfig object might not be ideal for direct logging.
    env_cfg_for_wandb = OmegaConf.to_container(base_env_cfg_obj, resolve=True)

    wandb.init(
        project="metta_vecenv_bench",  # New project for these benchmarks
        name=run_name,
        config={
            "env_config_name": args.env_config_name,
            "env_config_details": env_cfg_for_wandb,  # Log resolved base env config
            "total_individual_steps": args.total_steps,
            "log_interval_individual_steps": args.log_interval_steps,
            "steps_per_reset_cycle_individual": args.steps_per_reset_cycle,
            "vectorization_backend": args.vectorization,
            "num_environments": args.num_envs,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size or args.num_envs,
            "run_id": run_id,
        },
    )
    logging.info(f"Initiating VecEnv benchmark with wandb run name: {run_name}")

    logging.info("--- Running Continuous VecEnv Simulation Part ---")
    run_continuous_simulation(
        env_config=base_env_cfg_obj,
        vectorization=args.vectorization,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        log_interval_steps=args.log_interval_steps,
    )
    logging.info("--- Continuous VecEnv Simulation Part Finished ---")

    logging.info("--- Running VecEnv Simulation with Resets Part ---")
    run_simulation_with_resets(
        env_config=base_env_cfg_obj,
        vectorization=args.vectorization,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        log_interval_steps=args.log_interval_steps,
        steps_per_reset_cycle=args.steps_per_reset_cycle,
    )
    logging.info("--- VecEnv Simulation with Resets Part Finished ---")

    wandb.finish()
    logging.info("PufferLib VecEnv benchmark script finished.")


if __name__ == "__main__":
    main()
