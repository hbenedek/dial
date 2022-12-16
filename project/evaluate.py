from collections import deque
from env import DarpEnv
from log import logger, set_level
from generator import load_data, parse_aoyu
from utils import get_device
import torch
import model
from typing import Tuple, List
import pandas as pd
import time
import numpy as np

def simulate(env: DarpEnv, policy, greedy: bool=False) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    state = env.representation()
    done = False
    while not done:
        free_times = [vehicle.free_time for vehicle in env.vehicles]
        env.current_time = np.min(free_times)
        indices = np.argwhere(free_times == env.current_time)
        indices = indices.flatten().tolist()
        world, requests, vehicle  = state

        for vehicle_id in indices:
            state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
            mask = env.mask_illegal(vehicle_id)
            action, log_prob = policy.act(state, mask, greedy=greedy)
            #action = env.nearest_action_choice(vehicle_id)
            state, reward, done = env.step(action, vehicle_id)
            rewards.append(reward)
            log_prob = 0
            log_probs.append(log_prob)
    return rewards, log_probs


def evaluate_model(policy, env: DarpEnv, i: int=0, log: bool=True):
    #padding env to be able to use larger policy networks
    #if env.nb_requests < policy.nb_requests or env.nb_vehicles < policy.nb_vehicles:
    #    env.pad_env(policy.nb_vehicles, policy.nb_requests)
    #    logger.info("padding environment")

    #simulation
    rewards, _ = simulate(env, policy, greedy=True)
    env.penalize_broken_time_windows()

    #calculating metrics
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
    return total, env.penalty, delivered


def evaluate_aoyu(policy, instance: str):
    costs = []
    penalties = []
    tw_penalties = []
    rt_penalties = []
    rd_penalties = []
    objectives = []
    deliveries = []
    runtimes = []
    logger.info("parsing %s for testing data", instance)
    test_envs = parse_aoyu(instance)
    #policy.eval()
    for i,env in enumerate(test_envs):
        try:
            with torch.no_grad():
                start = time.time()
                cost, penalty, delivered = evaluate_model(policy, env, i=i ,log=False)
                end = time.time()
                logger.info("evaluating instance %s, OBJECTIVE: %s, PREDICT: %s", i, env.objective, cost)
            costs.append(cost)
            penalties.append(penalty["sum"])
            tw_penalties.append(penalty["tw_penalty"]) 
            rt_penalties.append(penalty["rt_penalty"]) 
            rd_penalties.append(penalty["rd_penalty"]) 
            deliveries.append(delivered)
            objectives.append(env.objective)
            runtime = end - start
            runtimes.append(runtime)
        except:
            logger.info("PROBLEM OCCURED DURING EVALUATING INSTANCE %s", i)
        if i > 99:
            break
        
    df = pd.DataFrame({"cost": costs, 
                        "penalty": penalties, 
                        "tw_penalties": tw_penalties, 
                        "rt_penalties": rt_penalties, 
                        "rd_penalties": rd_penalties ,
                        "objective": objectives,
                         "delivered": deliveries})
    df["gap"] = (df["cost"] / df["objective"] - 1) * 100
    mean_gap = df["gap"].mean()
    mean_penalty = df["penalty"].mean()
    mean_tw = df["tw_penalties"].mean()
    mean_rt = df["rt_penalties"].mean()
    logger.info("mean gap: %s, mean TW: %s, mean RT: %s, mean_penalty: %s", mean_gap, mean_tw, mean_rt, mean_penalty)
    return df


if __name__ == "__main__":
    import glob
    import re

    ####################    EXAMPLE USAGE 1 (EVALUATE MODEL PERFORMANCE ON 1 teste env)   ###########################
    logger = set_level(logger, "info")

    # loading the darp instance

    policy = model.Policy(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    # loading a Result object, containing a state_dict of a trained model (WARNING: for now model hyperparameters are not stored in the result object) 
    PATH = "models/new-resulta2-16-supervised-rf"
    r = load_data(PATH)
    state = r.policy_dict
    policy.load_state_dict(state)

    # passing the model to CUDA if available 
    device = get_device()
    policy.to(device)
    policy.eval()

    files = glob.glob("data/aoyu/[ab]*")
    instances = set([re.search("data/aoyu/(.*-.*)-.*", file).group(1) for file in files])
    instances = ["a2-16"]
    for instance in instances:
        FILE_NAME = f'data/cordeau/{instance}.txt'
        logger.info("EVALUATING %s", instance)
        nb_requests = int(re.search(".*-(.*)", instance).group(1))
        nb_vehicles = int(re.search("[ab](.*)-.*", instance).group(1))
        
        test_env = DarpEnv(size=10, nb_requests=nb_requests, nb_vehicles=nb_vehicles, time_end=1440, dataset=FILE_NAME)
        routing_cost, window_penalty, delivered = evaluate_model(policy, test_env)
        test_path = f"data/aoyu/{instance}-test.txt"
        #df = evaluate_aoyu(policy, test_path)
        #df.to_csv(f"evaluations/new-data-{instance}-baseline")

    ####################    EXAMPLE USAGE 2 (EVALUATE MODEL PERFORMANCE ON 1000 teste env)   ###########################

    # logger = set_level(logger, "info")

    # instance = "a2-16"
    # test_path = f"data/aoyu/{instance}-test.txt"

    # policy = model.Policy(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    # # # loading a Result object, containing a state_dict of a trained model (WARNING: for now model hyperparameters are not stored in the result object) 
    # PATH = "models/result-a2-16-supervised-rf-50-epochs"
    # r = load_data(PATH)
    # state = r.policy_dict
    # policy.load_state_dict(state)

    # # # passing the model to CUDA if available 
    # device = get_device()
    # policy.to(device)
    # policy.eval()

    # df = evaluate_aoyu(policy, test_path)
    # df.to_csv(f"evaluations/new-data-{instance}-test-model-nn-a2-16-05")


