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
        self.nb_tokens = nb_requests + 1 + 1
        #self.world_embedding = nn.Linear(8, d_model)
        #self.request_embedding = nn.Linear(10, d_model)
        #self.vehicle_embedding = nn.Linear(6, d_model)
        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoders = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.classifier = nn.Linear(self.nb_tokens * d_model, self.nb_actions) 
        self.device = get_device()
        self.to(self.device)

        # Request embeddings
        self.embed_time = nn.Embedding(time_end, d_model) #for all window constraints
        self.embed_position = nn.Embedding(env_size * 2 * 10, d_model)
        self.embed_request_id = nn.Embedding(nb_requests + 1, d_model)
        self.embed_request_status = nn.Embedding(3, d_model)

        # Vehicle embeddings 
        self.embed_vehicle_status = nn.Embedding(3, d_model)
        
        #World embeddings
        self.embed_current_vehicle = nn.Embedding(nb_vehicles, d_model)

    def embeddings(self, world, requests, vehicles):
        x = [self.embed_time(world[0].long().to(device))]
        self.embed_time(world[6].long().to(device))
        self.embed_time(world[7].long().to(device))
        x.append(self.embed_current_vehicle(world[1].long().to(device)))
        x.append(self.embed_position(world[2].long().to(device)))
        x.append(self.embed_position(world[3].long().to(device)))
        x.append(self.embed_position(world[4].long().to(device)))
        x.append(self.embed_position(world[5].long().to(device)))
        world = np.array([self.current_time, 
                    
                    self.max_ride_time,
                    self.max_route_duration])
    

    def forward(self, state):
        world, requests, vehicles = state 

        w, r, v = self.embeddings(world, requests, vehicles)

        world = world.to(self.device) # world.size() (bsz, nb_tokens, embed)
        requests = requests.to(self.device)
        vehicles = vehicles.to(self.device)
        #w = self.world_embedding(world)
        #r = self.request_embedding(requests)
        #v = self.vehicle_embedding(vehicles)
        
        x = torch.cat((w, r, v), dim=1) # x.size() = (bsz, nb_tokens, embed)
        x = x.permute([1,0,2]) # x.size() = (nb_tokens, bsz, embed)
        x = self.encoders(x)
        x = x.permute([1,0,2]).flatten(start_dim=1)
        x = self.classifier(x)
        return x
    
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
    state = [world.unsqueeze(0), requests.unsqueeze(0), vehicle.unsqueeze(0)]
    for t in range(max_step):
        mask = env.mask_illegal()
        if sum(mask) == 0:
            print("WARNING THIS SHIT CANNOT HAPPEN")
        if greedy:
            action, log_prob = policy.greedy(state, mask)
        else:
            action, log_prob = policy.act(state, mask)
        state, reward, done = env.step(action)
        world, requests, vehicle  = state
        state = [world.unsqueeze(0), requests.unsqueeze(0), vehicle.unsqueeze(0)]
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
             test_env: DarpEnv,
             relax_window: bool):
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
            #update baseline model after every 500 steps
            if i_episode % update_baseline == 0:
                if train_R >= baseline_R:
                    logger.info("new baseline model selected after achiving %s reward", train_R)
                    baseline.load_state_dict(policy.state_dict())

            env.reset(relax_window)
            baseline_env = copy.deepcopy(env)

            #simulate episode with train and baseline policy
            with torch.no_grad():
                baseline_rewards, _ = simulate(max_step, baseline_env, baseline, greedy=True)
                
            rewards, log_probs = simulate(max_step, env, policy)

            #aggregate rewards and logprobs
            train_R = sum(rewards)
            baseline_R = sum(baseline_rewards)
            #TODO: try nb_delivered as reward signal
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

if __name__ == "__main__":
    seed_everything(1)
    device = get_device() 
    FILE_NAME = 'data/cordeau/a2-16.txt'    
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=200, dataset=FILE_NAME)
    print([r for r in test_env.vehicles])

    path = "data/test_sets/generated-a2-16.pkl"
    envs = load_data(path)

    policy = Policy(d_model=128, nhead=4, nb_requests=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    logger = set_level(logger, "info")
    
    scores, tests = reinforce(policy=policy, 
                     optimizer=optimizer,
                     nb_epochs=10, 
                     update_baseline=100,
                     envs=envs,
                     test_env=test_env,
                     relax_window=False)
    #dump_data(scores, "models/scores.pkl")
    #dump_data(tests, "models/tests.pkl")
    PATH = "models/test.pth"
    torch.save(policy.state_dict(), PATH)
    # fig, ax = plt.subplots(1,1,figsize=(10,10))
    # ax.plot(scores)
    # ax.set(xlabel="Epsidoe", ylabel="Training Reward", title="Total distance")
    
    # plt.show()

    #TEST ENV WITH LOGS
    logger.info("FINAL TEST WITH LOGS")
    logger = set_level(logger, "debug")
    test_env.reset(relax_window=False)
    rewards, _ = simulate(100, test_env, policy)
    rewards = sum([r.state == "delivered" for r in test_env.requests])
    logger.info(f"total delivered: {rewards}")


