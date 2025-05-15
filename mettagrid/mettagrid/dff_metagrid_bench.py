import logging
import time

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    logging.info("Starting main function.")
    env_config_name = "benchmark"
    cfg = get_cfg(env_config_name)

    env = MettaGridEnv(env_cfg=cfg, render_mode=None)
    env.reset()
    logging.info("Environment reset.")
    count = 0
    last_log_time = time.monotonic()
    last_log_step_count = 0
    while True:
        action = env.action_space.sample()
        logging.debug(f"Step {count}, Action: {action}")
        env.step(action)
        count += 1

        current_time = time.monotonic()
        time_since_last_log = current_time - last_log_time

        if time_since_last_log >= 10.0:
            steps_this_interval = count - last_log_step_count
            sps = steps_this_interval / time_since_last_log
            logging.info(f"Steps per second: {sps:.2f} (over {time_since_last_log:.2f}s, {steps_this_interval} steps)")
            last_log_time = current_time
            last_log_step_count = count


if __name__ == "__main__":
    main()
