from env import DarpEnv
from log import logger, set_level
from generator import load_data, parse_aoyu
from utils import get_device
import torch
import model
from typing import Tuple, List
import pandas as pd

def simulate(max_step: int, env: DarpEnv, policy, greedy: bool=False) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    state = env.representation()
    for t in range(max_step):
        world, requests, vehicle  = state
        state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
        mask = env.mask_illegal()
        if greedy:
            action, log_prob = policy.greedy(state, mask)
        else:
            action, log_prob = policy.act(state, mask)
        state, reward, done = env.step(action)

        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            break 
    return rewards, log_probs


def evaluate_model(policy, env: DarpEnv, max_step: int=2000, i: int=0, log: bool=True):
    rewards, _ = simulate(max_step, env, policy, greedy=True)
    env.penalize_broken_time_windows()
    total = sum([v.total_distance_travelled for v in env.vehicles])
    delivered =  sum([request.state == "delivered" for request in env.requests])
    in_trunk = sum([r.state == "in_trunk" for r in env.requests])
    pickup = sum([r.state == "pickup" for r in env.requests])
    if log:
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
    return total, env.penalty["sum"]


def evaluate_aoyu(policy, instance: str):
    costs = []
    penalties = []
    objectives = []
    logger.info("parsing %s for testing data", instance)
    test_envs = parse_aoyu(instance)
    policy.eval()
    for i,env in enumerate(test_envs):
        try:
            max_step = env.nb_requests * 2 + env.nb_vehicles
            env.pad_env(policy.nb_vehicles, policy.nb_requests)
            with torch.no_grad():
                logger.info("evaluating instance %s", i)
                cost, penalty = evaluate_model(policy, env, max_step=max_step, i=i ,log=False)
            costs.append(cost)
            penalties.append(penalty)
            objectives.append(env.objective)
        except:
            logger.info("PROBLEM OCCURED DURING EVALUATING INSTANCE %s", i)
        
    df = pd.DataFrame({"cost": costs, "penalty": penalties, "objective": objectives})
    df["gap"] = (df["cost"] / df["objective"] - 1) * 100
    mean_gap = df["gap"].mean()
    mean_penalty = df["penalty"].mean()
    logger.info("mean gap: %s, mean_penalty: %s", mean_gap, mean_penalty)
    return df


if __name__ == "__main__":
   
    FILE_NAME = 'data/cordeau/a2-16.txt'
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=34, dataset=FILE_NAME)
    test_env.pad_env(4, 48)

    policy = model.Aoyu(d_model=256, nhead=8, nb_requests=48, nb_vehicles=4, num_layers=4, time_end=1440, env_size=10)
    PATH = "models/result-a4-48-supervised-rf-01-aoyu-model"
    #policy = model.Aoyu(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    #PATH = "models/result-a2-16-supervised-rf-02-aoyu512-model"
    state = torch.load(PATH)
    policy.load_state_dict(state)
    policy.eval()
    device = get_device()
    policy.to(device)
    #evaluate_model(policy, test_env, max_step=100)
    instance = "a2-16"

    df = evaluate_aoyu(policy, f"data/aoyu/{instance}-test.txt")
    df.to_csv("evaluations/data-a2-16-test-model-a4-48")

