from email.policy import Policy
import numpy as np
from collections import deque
from torch import double, optim
import torch.nn as nn
import torch
from utils import get_device
from typing import Tuple, List
from torch.distributions import Categorical
import numpy as np
from env import DarpEnv
from tqdm import tqdm 
from utils import coord2int, seed_everything
from generator import generate_environments, load_data, load_aoyo
import time
import copy
import matplotlib.pyplot as plt
from log import logger, set_level
import evaluate
from entity import Result, Vehicle
import gc
import psutil


#TODO: copy aoyu policy
class Policy(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, nb_requests: int=16, nb_vehicles: int=2, num_layers: int=2, time_end: int=1440, env_size: int=10):
        super(Policy, self).__init__()
        self.nb_actions = nb_requests + 1
        self.nb_tokens =  nb_requests 
        self.nb_vehicles = nb_vehicles
        self.nb_requests = nb_requests 
        self.device = get_device()
        self.time_end = time_end
        self.env_size = env_size
        self.d_model = d_model

        #Request embeddings
        self.embed_time = nn.Embedding(time_end * 10 + 1, d_model) 
        self.embed_position = nn.Embedding(env_size * 2 * 10 + 1, d_model)
        self.embed_request_id = nn.Embedding(nb_requests + 1, d_model)
        self.embed_request_status = nn.Embedding(3, d_model)
        self.embed_vehicle_status = nn.Embedding(3, d_model)
        self.embed_current_vehicle = nn.Embedding(nb_vehicles, d_model)
        self.embed_being_served = nn.Embedding(nb_vehicles + 1, d_model)

        #Transformers
        self.request_encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.request_linear = nn.Linear(15 * d_model, d_model)

        #Vehicle embedding
        #self.embed_vehicle = nn.Linear(nb_requests * 2 + 1, d_model)

        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoders = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.classifier = nn.Linear(self.nb_tokens * d_model, self.nb_actions)

    def __repr__(self):
        return f"policy_{self.d_model}"

    def __str__(self):
        return f"policy_{self.d_model}"

    def embeddings(self, world, requests, vehicles):
        r = []
        world = world.float()
        requests = requests.float()
        vehicles = vehicles.float()
        world = torch.transpose(world, 0, 1) #world.size() = (8, bsz)
        vehicles = torch.transpose(vehicles, 0, 1) #vehicles.size() = (6, bsz)
        zeros = torch.zeros(world[1].size())
        time_max = torch.ones(world[1].size()) * self.time_end
        result = []
        #requests.size() = (bsz, nb_requests, 10)
        for i in range(self.nb_requests):
            #position of current vehicle
            r = []
            r.append(self.embed_position(torch.round(vehicles[0] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(vehicles[1] * 10 + self.env_size * 10).long().to(self.device)))

            #max ride time and max route duration
            r.append(self.embed_time(torch.round(10 * world[6]).long().to(self.device)))
            r.append(self.embed_time(torch.round(10 * world[7]).long().to(self.device))) 
            r.append(self.embed_request_id(requests[:,i,0].long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,1] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,2] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,3] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,4] * 10 + self.env_size * 10).long().to(self.device)))
            shifted5 = torch.minimum(torch.maximum(requests[:,i,5] - world[1], zeros), time_max)
            shifted6 = torch.minimum(torch.maximum(requests[:,i,6] - world[1], zeros), time_max)
            shifted7 = torch.minimum(torch.maximum(requests[:,i,7] - world[1], zeros), time_max)
            shifted8 = torch.minimum(torch.maximum(requests[:,i,8] - world[1], zeros), time_max)
            r.append(self.embed_time(torch.round(shifted5 * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(shifted6 * 10).long().to(self.device))) 
            r.append(self.embed_time(torch.round(shifted7 * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(shifted8 * 10).long().to(self.device)))
            r.append(self.embed_request_status(requests[:,i,9].long().to(self.device)))
            r.append(self.embed_being_served(requests[:,i,10].long().to(self.device)))
            r = torch.stack(r).transpose(0, 1) #r.size() = (bsz, 15, d_model)
            r = self.request_encoder(r) 
            r = self.request_linear(r.flatten(start_dim=1))
            #x = vehicles[2:].float().to(self.device).transpose(0, 1)
            #r.append(self.embed_vehicle(x))
            result.append(r)
        result = torch.stack(result).transpose(0, 1)
        return result

    def forward(self, state):
        world, requests, vehicles = state 
        r = self.embeddings(world, requests, vehicles)
        r = r.permute([1,0,2]) #x.size() = (nb_tokens, bsz, embed)
        r = self.encoders(r)
        r = r.permute([1,0,2]).flatten(start_dim=1)
        r = self.classifier(r)
        return r  #x.size() = (bsz, nb_actions)

    def act(self, state, mask, greedy=False):
        out = self.forward(state).cpu()
        out = out * mask
        mask = mask.type(torch.BoolTensor)
        out = torch.where(mask, out, torch.tensor(-1e+8).double())
        probs = nn.Softmax(dim=1)(out)
        m = Categorical(probs)
        if greedy:
            action = probs.max(1)[1].view(1, 1)
        else:
            action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_episodes,
             nb_requests: int,
             nb_vehicles: int,
             update_baseline: int,
             test_env: DarpEnv):

    env = DarpEnv(10, nb_requests, nb_vehicles, time_end=1440, max_step=nb_requests * 2 + nb_vehicles, max_route_duration=480, capacity=3, max_ride_time=30, dataset=PATH)         
    #init baseline model
    baseline = copy.deepcopy(policy)
    baseline = baseline.to(device)
    policy = policy.to(device)
    train_R = 0
    baseline_R = 0
    routes = []
    penalties = []
    train_losses = []
    baseline_losses = []


    for i_episode in range(nb_episodes):
        logger.info("episode: %s, RAM: %s, CPU: %s, LOSS %s", i_episode, psutil.virtual_memory().percent, psutil.cpu_percent(), train_R)
        
        #update baseline model after every n steps
        if i_episode % update_baseline == 0:
            if sum(train_losses) <= sum(baseline_losses):
                logger.info("new baseline model selected after achiving %s reward", train_R)
                baseline.load_state_dict(policy.state_dict())
                train_losses.clear()
                baseline_losses.clear()

        env.reset()
        entities = env.vehicles, env.requests, env.depots
        #simulate episode with train and baseline policy
        with torch.no_grad():
            baseline_rewards, _ = evaluate.simulate(env, baseline, greedy=True)
            env.penalize_broken_time_windows()
            baseline_penalty = env.penalty["sum"] 
            env.reset(entities=entities)  
            
        rewards, log_probs = evaluate.simulate(env, policy)
        env.penalize_broken_time_windows()
        penalty = env.penalty["sum"] 

        #aggregate rewards and logprobs
        train_R = penalty + sum(rewards) 
        baseline_R = baseline_penalty + sum(baseline_rewards) 
        sum_log_probs = sum(log_probs)

        train_losses.append(train_R)
        baseline_losses.append(baseline_R)

        policy_loss = torch.mean(-train_R * sum_log_probs)
    
        #backpropagate
        optimizer.zero_grad()
        policy_loss.backward()
        train_losses.append(policy_loss)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
        optimizer.step()

        #garbage collector
        gc.collect()

        #test phase
        if i_episode % 100 == 0:
            with torch.no_grad():
                policy.eval()
                test_env.reset()
                route, penalty, delivered = evaluate.evaluate_model(policy, test_env, i=i_episode)
                routes.append(route)
                penalties.append(penalty["sum"])
                policy.train()

    #saving results
    policy = policy.to("cpu")
    state_dict = policy.state_dict()

    result = Result(id)
    result.routes = routes
    result.penalty = penalties
    result.train_loss = train_losses
    result.test_loss = 0
    result.policy_dict = state_dict
    return result


def reinforce_trainer(test_env_path: str, 
                    result_path: str,
                    id: str,
                    nb_episodes: int, 
                    nb_requests: int,
                    nb_vehicles: int,
                    update_baseline: int,
                    policy: Policy, 
                    optimizer: torch.optim.Optimizer):
           
    test_env = DarpEnv(10, 
                        nb_requests, 
                        nb_vehicles, 
                        time_end=1440, 
                        max_step=nb_requests * 2 + nb_vehicles, 
                        dataset=test_env_path)
    logger.info("dataset successfully loaded")

    logger.info("training starts")
    result = reinforce(policy, optimizer, nb_episodes, nb_requests, nb_vehicles,  update_baseline, test_env=test_env)
    torch.save(policy.state_dict(), result_path + '/' + id + "-model")
    return result


if __name__ == "__main__":
    seed_everything(1)
    result_path = "models"
    nb_vehicles = 2
    nb_requests = 16
    trial = "03"
    variant = "a"
    instance = f"{variant}{nb_vehicles}-{nb_requests}"
    test_env_path = f'data/cordeau/{instance}.txt'  
    PATH = test_env_path
    id = f"result-{instance}-reinforce-{trial}-aoyu"

    nb_episodes= 2000
    update_baseline = 20

    policy = Policy(d_model=256, nhead=8, nb_requests=nb_requests, nb_vehicles=nb_vehicles, num_layers=4, time_end=1440, env_size=10)
    device = get_device()
    model_path = "models/result-a2-16-supervised-nn-01-aoyu256"
    r = load_data(model_path)
    state = r.policy_dict
    policy.load_state_dict(state)
    logger.info("training on device: %s", device)
    policy = policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-3)

    logger = set_level(logger, "info")
    reinforce_trainer(test_env_path, result_path, id ,nb_episodes, nb_requests, nb_vehicles, update_baseline, policy, optimizer)
  
    


