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
from utils import coord2int, init_logger
from data import generate_training_data, dump_training_data, load_training_data
import time
import copy
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, d_model=512, nhead=8, nb_actions=10, nb_tokens=4):
        super(Policy, self).__init__()
        self.world_embedding = nn.Linear(4, d_model)
        self.request_embedding = nn.Linear(11, d_model)
        self.vehicle_embedding = nn.Linear(7, d_model)
        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(nb_tokens * d_model, nb_actions) 

    def forward(self, world, requests, vehicles): 
        w = self.world_embedding(world)
        r = self.request_embedding(requests)
        v = self.vehicle_embedding(vehicles)
        x = torch.cat((w,r, v), dim=0).unsqueeze(dim=1) # x.size() = (nb_tokens, bsz, embed)
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
        action, log_prob = policy.act(state, mask)
        state, reward, done = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        if done:
            break 
    return rewards, log_probs


#TODO: change this
def reinforce(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int, 
             max_time: int, 
             update_baseline: int,
             envs: List[DarpEnv],
             test_env: DarpEnv):
    #init baseline model
    baseline = copy.deepcopy(policy)
    scores = []

    for i_epoch in range(nb_epochs):
        for i_episode, env in tqdm(enumerate(envs)):
            #update baseline model after every 500 steps
            if i_episode %  update_baseline == 0:
                baseline.load_state_dict(policy.state_dict())
            
            env.reset()
            baseline_env = copy.deepcopy(env)

            #simulate episode with train and baseline policy
            with torch.no_grad():
                baseline_rewards, _ = simulate(max_time, baseline_env, baseline)
            rewards, log_probs = simulate(max_time, env, policy)

            #aggregate rewards and logprobs
            train_R = sum(rewards)
            baseline_R = sum(baseline_rewards)
            sum_log_probs = sum(log_probs)
            scores.append(train_R)

            policy_loss = torch.mean((train_R - baseline_R) * sum_log_probs)
        
            #backpropagate
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            if i_episode % 100 == 0:
                test_env.reset()
                rewards, log_probs = simulate(max_time, test_env, policy)
                print('Episode {}\tTotal distance: {:.2f}'.format(i_episode, sum(rewards)))
            
    return scores

if __name__ == "__main__":
    device = get_device() #TODO: pass everything to device
    FILE_NAME = '../data/cordeau/a2-16.txt'    
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=1000, dataset=FILE_NAME)
    
    path = "../data/test_sets/generated-a2-16.pkl"
    envs = load_training_data(path)

    nb_actions = test_env.nb_requests * 2 + 1
    nb_tokens = test_env.nb_requests + test_env.nb_vehicles + 1

    policy = Policy(d_model=128, nhead=4, nb_actions=nb_actions, nb_tokens=nb_tokens)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)


    scores = reinforce(policy=policy, 
                    optimizer=optimizer,
                    nb_epochs=1, 
                    max_time=1400, 
                    update_baseline=100,
                    envs=envs,
                    test_env=test_env)

    plt.plot(scores)
    plt.show()

    #TEST ENV WITH LOGS
    logger = init_logger(level="debug") #this makes a second logger, somehow need to figure out how to change log level to debug
    test_env.reset()
    rewards, _ = simulate(1000, test_env, policy)
    #TODO: look at logs, for some reasone getting 0 cumrewards still
    print(f"Episode finished with reward {sum(rewards)}")

