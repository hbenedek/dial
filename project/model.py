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
from generator import load_data, dump_data
import time
import copy
import matplotlib.pyplot as plt
from log import logger, set_level

class Policy(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, nb_requests: int=16, nb_vehicles: int=2, num_layers: int=2, time_end: int=1400, env_size: int=10):
        super(Policy, self).__init__()
        self.nb_actions = nb_requests * 2 + 1
        self.nb_tokens = 8 + nb_requests * 10 + 6
        self.nb_requests = nb_requests 
        self.device = get_device()
        self.time_end = time_end
        self.env_size = env_size
        
        #Transformers
        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoders = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.classifier = nn.Linear(self.nb_tokens * d_model, self.nb_actions) 


        #Request embeddings
        self.embed_time = nn.Embedding(time_end * 10 + 1, d_model) 
        self.embed_position = nn.Embedding(env_size * 2 * 10 + 1, d_model)
        self.embed_request_id = nn.Embedding(nb_requests + 1, d_model)
        self.embed_request_status = nn.Embedding(3, d_model)

        #Vehicle embeddings 
        self.embed_vehicle_status = nn.Embedding(3, d_model)
        
        #World embeddings
        self.embed_current_vehicle = nn.Embedding(nb_vehicles, d_model)

    def embeddings(self, world, requests, vehicles):
        world = torch.transpose(world, 0, 1) #world.size() = (8, bsz)
        w = [self.embed_time(torch.round(10 * world[0]).long().to(self.device))]
        w.append(self.embed_current_vehicle(world[1].long().to(self.device)))
        w.append(self.embed_position(torch.round(world[2] * 10 + self.env_size * 10).long().to(self.device)))
        w.append(self.embed_position(torch.round(world[3] * 10 + self.env_size * 10).long().to(self.device)))
        w.append(self.embed_position(torch.round(world[4] * 10 + self.env_size * 10).long().to(self.device)))
        w.append(self.embed_position(torch.round(world[5] * 10 + self.env_size * 10).long().to(self.device)))
        w.append(self.embed_time(torch.round(10 * world[6]).long().to(self.device)))
        w.append(self.embed_time(torch.round(10 * world[7]).long().to(self.device)))
        w = torch.stack(w).transpose(0, 1) #w.size() = (bsz, 8, d_model)

        r = []
        #requests.size() = (bsz, nb_requests, 10)
        for i in range(self.nb_requests):
            r.append(self.embed_request_id(requests[:,i,0].long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,1] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,2] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,3] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,4] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(requests[:,i,5] * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(requests[:,i,6] * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(requests[:,i,7] * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(requests[:,i,8] * 10).long().to(self.device)))
            r.append(self.embed_request_status(requests[:,i,9].long().to(self.device)))
        r = torch.stack(r).transpose(0, 1) #r.size() = (bsz, 10 * nb_requests, d_model)
       
        vehicles = torch.transpose(vehicles, 0, 1) #vehicles.size() = (6, bsz)
        v = []
        v.append(self.embed_position(torch.round(vehicles[0] * 10 + self.env_size * 10).long().to(self.device)))
        v.append(self.embed_position(torch.round(vehicles[1] * 10 + self.env_size * 10).long().to(self.device)))
        v.append(self.embed_vehicle_status(vehicles[2].long().to(self.device)))
        v.append(self.embed_request_id(vehicles[3].long().to(self.device)))
        v.append(self.embed_request_id(vehicles[4].long().to(self.device)))
        v.append(self.embed_request_id(vehicles[5].long().to(self.device)))
        v = torch.stack(v).transpose(0, 1) #v.size() = (bsz, 6, d_model)

        return w, r, v

    

    def forward(self, state):
        world, requests, vehicles = state 
        w, r, v = self.embeddings(world, requests, vehicles)
        
        x = torch.cat((w, r, v), dim=1) #x.size() = (bsz, nb_tokens, embed)
        x = x.permute([1,0,2]) #x.size() = (nb_tokens, bsz, embed)
        x = self.encoders(x)
        x = x.permute([1,0,2]).flatten(start_dim=1)
        x = self.classifier(x)
        return x  #x.size() = (bsz, nb_actions)
    
    def act(self, state, mask):
        out = self.forward(state).cpu()
        out = out * mask
        probs = nn.Softmax(dim=1)(out)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def greedy(self, state, mask):
        out = self.forward(state).cpu()
        out = out * mask
        probs = nn.Softmax(dim=1)(out)
        m = Categorical(probs)
        action = probs.max(1)[1].view(1, 1)
        return action.item(), m.log_prob(action)


def simulate(max_step: int, env: DarpEnv, policy: Policy, greedy: bool=False) -> Tuple[List[float], List[float]]:
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


def reinforce(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int, 
             max_step: int, 
             update_baseline: int,
             envs: List[DarpEnv],
             test_env: DarpEnv):
    #init baseline model
    baseline = copy.deepcopy(policy)
    baseline = baseline.to(device)
    policy = policy.to(device)
    scores = []
    tests = []
    train_R = 0
    baseline_R = 0
    for i_epoch in range(nb_epochs):
        logger.info("*** EPOCH %s ***", i_epoch)
        for i_episode, env in enumerate(envs):
            print(i_episode)
            #update baseline model after every 500 steps
            #if i_episode % update_baseline == 0:
            #    if train_R >= baseline_R:
            #        logger.info("new baseline model selected after achiving %s reward", train_R)
            #        baseline.load_state_dict(policy.state_dict())

            env.reset()
            #baseline_env = copy.deepcopy(env)

            #simulate episode with train and baseline policy
            #with torch.no_grad():
            #    baseline_rewards, _ = simulate(max_step, baseline_env, baseline, greedy=True)
                
            rewards, log_probs = simulate(max_step, env, policy)

            #aggregate rewards and logprobs
            train_R = sum(rewards)
            #baseline_R = sum(baseline_rewards)
            sum_log_probs = sum(log_probs)
            scores.append(train_R)

            policy_loss = torch.mean(-train_R * sum_log_probs)
        
            #backpropagate
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
            if i_episode % 100 == 0:
                test_env.reset()
                with torch.no_grad():
                    rewards, log_probs = simulate(max_step, test_env, policy, greedy=True)

                delivered = sum([r.state == "delivered" for r in test_env.requests])
                logger.info('Episode: {}, total distance: {:.2f}'.format(i_episode, sum(rewards)))
                tests.append((sum(rewards), delivered))
                baseline.load_state_dict(policy.state_dict()) #update baseline
                in_trunk = sum([r.state == "in_trunk" for r in test_env.requests])
                pickup = sum([r.state == "pickup" for r in test_env.requests])
                logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
            
    return scores, tests

def reinforce_trainer(envs_path: str):
    envs = load_data(path)
    pass


if __name__ == "__main__":
    seed_everything(1)
    device = get_device() 
    FILE_NAME = 'data/cordeau/a2-16.txt'    
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=200, dataset=FILE_NAME)

    path = "data/processed/generated-a2-16.pkl"
    envs = load_data(path)

    policy = Policy(d_model=128, nhead=4, nb_requests=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    logger = set_level(logger, "info")
    
    scores, tests = reinforce(policy=policy, 
                      optimizer=optimizer,
                      nb_epochs=10, 
                      max_step = 100,
                      update_baseline=100,
                      envs=envs,
                      test_env=test_env)
    # #dump_data(scores, "models/scores.pkl")
    # #dump_data(tests, "models/tests.pkl")
    # PATH = "models/test.pth"
    # torch.save(policy.state_dict(), PATH)
    # # fig, ax = plt.subplots(1,1,figsize=(10,10))
    # # ax.plot(scores)
    # # ax.set(xlabel="Epsidoe", ylabel="Training Reward", title="Total distance")
    
    # # plt.show()

    # #TEST ENV WITH LOGS
    # logger.info("FINAL TEST WITH LOGS")
    # logger = set_level(logger, "debug")
    # test_env.reset(relax_window=False)
    # rewards, _ = simulate(100, test_env, policy)
    # rewards = sum([r.state == "delivered" for r in test_env.requests])
    # logger.info(f"total delivered: {rewards}")



