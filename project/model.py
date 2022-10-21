import numpy as np
from collections import deque
from torch import optim
import torch.nn as nn
import torch
from utils import get_device
from typing import Tuple, List
from torch.distributions import Categorical
import numpy as np
from env import DarpEnv
from tqdm import tqdm 
from utils import coord2int, seed_everything
from data import generate_training_data, dump_training_data, load_training_data
import time
import copy
import matplotlib.pyplot as plt
from log import logger, set_level

class Policy(nn.Module):
    def __init__(self, d_model=512, nhead=8, nb_actions=10, nb_tokens=4):
        super(Policy, self).__init__()
        self.world_embedding = nn.Linear(4, d_model)
        self.request_embedding = nn.Linear(11, d_model)
        self.vehicle_embedding = nn.Linear(7, d_model)
        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(nb_tokens * d_model, nb_actions) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, world, requests, vehicles): 
        w = self.world_embedding(world)
        r = self.request_embedding(requests)
        v = self.vehicle_embedding(vehicles)
        x = torch.cat((w,r, v), dim=0).unsqueeze(dim=1) # x.size() = (nb_tokens, bsz, embed)
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.permute([1,0,2]).flatten(start_dim=1)
        x = self.classifier(x)
        return self.softmax(x)
    
    def act(self, state, mask):
        world, requests, vehicles, = state 
        probs = self.forward(world, requests, vehicles).cpu()
        probs = probs * mask
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def simulate(max_step: int, env: DarpEnv, policy: Policy) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    state = env.representation()
    for t in range(max_step):
        mask = env.mask_illegal()
        if sum(mask) == 0:
            print("WARNING THIS SHIT CANNOT HAPPEN")
        action, log_prob = policy.act(state, mask)
        state, reward, done = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            break 
    return rewards, log_probs


def reinforce(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int, 
             max_time: int, 
             update_baseline: int,
             envs: List[DarpEnv],
             test_env: DarpEnv,
             relax_window: bool = False):
    #init baseline model
    baseline = copy.deepcopy(policy)
    scores = []
    train_R = 0
    baseline_R = 0
    for i_epoch in range(nb_epochs):
        for i_episode, env in tqdm(enumerate(envs)):
            #update baseline model after every 500 steps
            if i_episode % update_baseline == 0:
                if train_R >= baseline_R:
                    logger.info("new baseline model selected after achiving %s reward", train_R)
                    baseline.load_state_dict(policy.state_dict())

            env.reset(relax_window)
            baseline_env = copy.deepcopy(env)

            #simulate episode with train and baseline policy
            with torch.no_grad():
                baseline_rewards, _ = simulate(max_time, baseline_env, baseline)
                
            rewards, log_probs = simulate(max_time, env, policy)

            #aggregate rewards and logprobs
            #train_R = sum(rewards)
            #baseline_R = sum(baseline_rewards)
            #TODO: try nb_delivered as reward signal
            train_R = sum([r.state == "delivered" for r in env.requests])
            baseline_R = sum([r.state == "delivered" for r in baseline_env.requests])
            sum_log_probs = sum(log_probs)
            scores.append(train_R)

            policy_loss = torch.mean(- (train_R - baseline_R) * sum_log_probs)
        
            #backpropagate
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
            
            if i_episode % 100 == 0:
                test_env.reset(relax_window)
                rewards, log_probs = simulate(max_time, test_env, policy)
                rewards = sum([r.state == "delivered" for r in test_env.requests])
                logger.info('Episode {}: delivered requests: {:.2f}'.format(i_episode, rewards))
                if rewards == test_env.nb_requests:
                    break
                #delivered = sum([r.state == "delivered" for r in test_env.requests])
                #in_trunk = sum([r.state == "in_trunk" for r in test_env.requests])
                #pickup = sum([r.state == "pickup" for r in test_env.requests])
                #print(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
            
    return scores

if __name__ == "__main__":
    seed_everything(1)
    device = get_device() #TODO: pass everything to device
    FILE_NAME = '../data/cordeau/a2-16.txt'    
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=1000, dataset=FILE_NAME)
    
    path = "../data/test_sets/generated-a2-16.pkl"
    envs = load_training_data(path)

    nb_actions = test_env.nb_requests * 2 + 1
    nb_tokens = test_env.nb_requests + 1 + 1

    policy = Policy(d_model=100, nhead=4, nb_actions=nb_actions, nb_tokens=nb_tokens)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    logger = set_level(logger, "info")
    
    scores = reinforce(policy=policy, 
                    optimizer=optimizer,
                    nb_epochs=1, 
                    max_time=1400, 
                    update_baseline=50,
                    envs=envs,
                    test_env=test_env,
                    relax_window=True)

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(scores)
    ax.set(xlabel="Epsidoe", ylabel="Training Reward", title="Total distance")
    
    plt.show()

    #TEST ENV WITH LOGS
    logger = set_level(logger, "debug")
    test_env.reset(relax_window=True)
    rewards, _ = simulate(1400, test_env, policy)
    rewards = sum([r.state == "delivered" for r in test_env.requests])
    print(f"total delivered: {rewards}")


