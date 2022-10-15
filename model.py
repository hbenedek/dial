import numpy as np
from requests import request
import torch.nn as nn
import torch
from utils import get_device
from torch.distributions import Categorical

import numpy as np
from env import DarpEnv
from utils import coord2int, init_logger
import time


def representation(env: DarpEnv) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    world = np.array([env.current_time, env.current_vehicle, coord2int(env.start_depot[1]), coord2int(env.end_depot[1])])
    requests = np.stack([r.get_vector() for r in env.requests])
    vehicles = np.stack([v.get_vector() for v in env.vehicles])
    w_tensor = torch.from_numpy(world).type(torch.FloatTensor).unsqueeze(dim=0)
    r_tensor = torch.from_numpy(requests).type(torch.FloatTensor)
    v_tensor = torch.from_numpy(vehicles).type(torch.FloatTensor)
    return w_tensor, r_tensor, v_tensor


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
    
    #def act(self, state):
    #    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    #    probs = self.forward(state).cpu()
    #    m = Categorical(probs)
    #    action = m.sample()
    #    return action.item(), m.log_prob(action)


#TODO: change this
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        ## Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## We calculate the return by sum(gamma[t] * reward[t]) 
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
    return scores

if __name__ == "__main__":
    device = get_device()
    logger = init_logger()
    FILE_NAME = './data/test_sets/t1-2.txt'    
    env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, dataset=FILE_NAME, logger=logger)
    world, requests, vehicles = representation(env)

    model = Policy(256, 8)
    probs = model(world, requests, vehicles)
    print(probs.size())

