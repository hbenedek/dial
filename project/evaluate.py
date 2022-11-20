from env import DarpEnv
from log import logger, set_level
from generator import load_data
from utils import get_device
import torch
import model
from typing import Tuple, List

def simulate(max_step: int, env: DarpEnv, policy, greedy: bool=False) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    world, requests, vehicle = env.representation()
    state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
    for t in range(max_step):
        mask = env.mask_illegal()
        if greedy:
            action, log_prob = policy.greedy(state, mask)
        else:
            action, log_prob = policy.act(state, mask)
        state, reward, done = env.step(action)
        world, requests, vehicle  = state
        state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            break 
    return rewards, log_probs


def evaluate_model(policy, env: DarpEnv, max_step: int=2000, i=0):
    rewards, _ = simulate(max_step, env, policy, greedy=True)
    env.penalize_broken_time_windows()
    total = sum([v.total_distance_travelled for v in env.vehicles])
    delivered =  sum([request.state == "delivered" for request in env.requests])
    in_trunk = sum([r.state == "in_trunk" for r in env.requests])
    pickup = sum([r.state == "pickup" for r in env.requests])
    logger.info(f"TEST {i}*****************************************************************")
    logger.info(f"reward: {total}")
    for vehicle in env.vehicles:
        logger.info(f'{vehicle} history: {vehicle.history}')
    logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
    logger.info("*** PENALTY ***")
    logger.info("start_window: %s", env.penalty["start_window"])
    logger.info("end_window: %s", env.penalty["end_window"])
    logger.info("max_route_duration: %s", env.penalty["max_route_duration"])
    logger.info("max_ride_time: %s", env.penalty["max_ride_time"])
    logger.info("sum: %s", env.penalty["sum"])
    logger.info(f"TEST {i}*****************************************************************")

if __name__ == "__main__":
    logger = set_level(logger, "debug")

    FILE_NAME = 'data/cordeau/a2-16.txt'
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=34, dataset=FILE_NAME)

    policy = model.Aoyu(d_model=128, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)

    #policy = Policy(d_model=128, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    PATH = "models/result-a2-16-supervised-rf-01-aoyu-model"
    state = torch.load(PATH)
    policy.load_state_dict(state)
    policy.eval()
    device = get_device()
    policy.to(device)
    
    evaluate_model(policy, test_env, max_step=200)