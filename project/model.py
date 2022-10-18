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
import time
import copy
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, d_model=512, nhead=8, nb_actions = 10, nb_tokens = 4):
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
             nb_episodes: int, 
             max_time: int, 
             print_every: bool,
             batch: int,
             env: DarpEnv,
             test_env: DarpEnv):
    #init baseline model
    baseline = copy.deepcopy(policy)
    scores = []
    for i_episode in tqdm(range(0, nb_episodes)):
        #update baseline model after every 500 steps
        if i_episode % 500 == 0:
            baseline.load_state_dict(policy.state_dict())
          
        #generate new env
        #TODO: maybe generate new env instance after every x simulations
        env.reset()
        baseline_env = copy.deepcopy(env)

        #simulate episode with train  and baseline policy
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
        
        if i_episode % print_every == 0:
            test_env.reset()
            rewards, log_probs = simulate(max_time, test_env, policy)
            cum_reward = sum(rewards)
            print('Episode {}\tTotal distance: {:.2f}'.format(i_episode, cum_reward))
        
    return scores

if __name__ == "__main__":
    device = get_device()
    FILE_NAME = '../data/test_sets/t1-2.txt'    
    test_env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, dataset=FILE_NAME)
    env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, capacity=3)
    policy = Policy(d_model=128, nhead=4, nb_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scores = reinforce(policy=policy, optimizer=optimizer, nb_episodes=1000, max_time=1400, print_every=100, env=env, test_env=test_env, batch=20)

    plt.plot(scores)
    plt.show()

    # TEST ENV WITH LOGS
    logger = init_logger(level="debug")
    test_env.reset()
    rewards, _ = simulate(100, test_env, policy)
    R = sum(rewards)
    print(f"Episode finished with reward {R}")

