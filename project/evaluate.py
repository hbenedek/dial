from collections import deque
from env import DarpEnv
from log import logger, set_level
from generator import load_data, parse_aoyu
from utils import get_device
import torch
import model
from typing import Tuple, List
import pandas as pd

def simulate(env: DarpEnv, policy, greedy: bool=False) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    state = env.representation()
    #env.vehicles[0].routes =  deque([9, 10, 9, 10, 6, 6, 1, 1, 5, 5, 13, 7, 13, 7, 14, 14, 4, 4, 17])
    #env.vehicles[1].routes = deque([2, 2, 8, 16, 8, 16, 3, 3, 12, 12, 15, 15, 11, 11, 17])
    for t in range(env.max_step):
        world, requests, vehicle  = state
        state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
        mask = env.mask_illegal()
        #action = env.vehicles[env.current_vehicle].routes.popleft()-1
        action, log_prob = policy.act(state, mask, greedy=greedy)
        state, reward, done = env.step(action)

        rewards.append(reward)
        #log_probs.append(log_prob)
        if done:
            break 
    return rewards, log_probs


def evaluate_model(policy, env: DarpEnv, i: int=0, log: bool=True):
    #padding env to be able to use larger policy networks
    if env.nb_requests < policy.nb_requests or env.nb_vehicles < policy.nb_vehicles:
        env.pad_env(policy.nb_vehicles, policy.nb_requests)
        logger.debug("padding environment")

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
    nb_penalties = []
    objectives = []
    deliveries = []
    logger.info("parsing %s for testing data", instance)
    test_envs = parse_aoyu(instance)
    policy.eval()
    for i,env in enumerate(test_envs):
        try:
            with torch.no_grad():
                logger.info("evaluating instance %s", i)
                cost, penalty, delivered = evaluate_model(policy, env, i=i ,log=False)
            costs.append(cost)
            penalties.append(penalty["sum"])
            nb_penalties.append(penalty["nb_penalty"]) 
            deliveries.append(delivered)
            objectives.append(env.objective)
        except:
            logger.info("PROBLEM OCCURED DURING EVALUATING INSTANCE %s", i)
        
    df = pd.DataFrame({"cost": costs, "penalty": penalties, "nb_penalties": nb_penalties ,"objective": objectives, "delivered": deliveries})
    df["gap"] = (df["cost"] / df["objective"] - 1) * 100
    mean_gap = df["gap"].mean()
    mean_penalty = df["penalty"].mean()
    logger.info("mean gap: %s, mean_penalty: %s", mean_gap, mean_penalty)
    return df


if __name__ == "__main__":

    ####################    EXAMPLE USAGE 1 (EVALUATE MODEL PERFORMANCE ON 1 teste env)   ###########################
    logger = set_level(logger, "debug")

    # loading the darp instance
    FILE_NAME = 'data/cordeau/b2-16.txt'
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=34, dataset=FILE_NAME)

    policy = model.Policy(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    # loading a Result object, containing a state_dict of a trained model (WARNING: for now model hyperparameters are not stored in the result object) 
    PATH = "models/result-a2-16-supervised-rf-05-aoy4layer"
    r = load_data(PATH)
    state = r.policy_dict
    policy.load_state_dict(state)

    # passing the model to CUDA if available 
    device = get_device()
    policy.to(device)
    policy.eval()

    routing_cost, window_penalty, delivered = evaluate_model(policy, test_env)

    ####################    EXAMPLE USAGE 2 (EVALUATE MODEL PERFORMANCE ON 1000 teste env)   ###########################

    #logger = set_level(logger, "info")

    #instance = "a2-16"
    #test_path = f"data/aoyu/{instance}-test.txt"

    # policy = model.Aoyu(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    # # loading a Result object, containing a state_dict of a trained model (WARNING: for now model hyperparameters are not stored in the result object) 
    # PATH = "models/result-a2-16-supervised-rf-01-aoyu256"
    # r = load_data(PATH)
    # state = r.policy_dict
    # policy.load_state_dict(state)

    # # passing the model to CUDA if available 
    # device = get_device()
    # policy.to(device)
    # policy.eval()

    #df = evaluate_aoyu(policy, test_path)
    #df.to_csv(f"evaluations/new-data-{instance}-test-model-nn-a2-16-05")


