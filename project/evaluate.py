from pathlib import Path
from model import Policy, Aoyu
from env import DarpEnv
from log import logger, set_level
from generator import load_data
from model import simulate
from utils import get_device
import torch


def evaluate_model(policy: Policy, env: DarpEnv, max_step: int=2000):
    env.reset()
    rewards, _ = simulate(max_step, env, policy, greedy=True)
    env.penalize_broken_time_windows()
    total = sum([v.total_distance_travelled for v in env.vehicles])
    delivered =  sum([request.state == "delivered" for request in env.requests])
    in_trunk = sum([r.state == "in_trunk" for r in env.requests])
    pickup = sum([r.state == "pickup" for r in env.requests])
    logger.info(f"reward: {total}")
    logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
    logger.info("*** PENALTY ***")
    logger.info("start_window: %s", env.penalty["start_window"])
    logger.info("end_window: %s", env.penalty["end_window"])
    logger.info("max_route_duration: %s", env.penalty["max_route_duration"])
    logger.info("max_ride_time: %s", env.penalty["max_ride_time"])
    logger.info("sum: %s", env.penalty["sum"])

if __name__ == "__main__":
    logger = set_level(logger, "debug")

    FILE_NAME = 'data/cordeau/a2-16.txt'
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=1000, dataset=FILE_NAME)
    policy = Aoyu(d_model=128, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)

    #policy = Policy(d_model=128, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    PATH = "models/result-a2-16-supervised-nn-08-aoyu-model"
    state = torch.load(PATH)
    policy.load_state_dict(state)
    policy.eval()
    device = get_device()
    policy.to(device)

    evaluate_model(policy, test_env, max_step=200)
