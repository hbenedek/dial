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
from entity import Result
import gc
import psutil

class DecoderModel(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, nb_requests: int=16, nb_vehicles: int=2, num_layers: int=2, time_end: int=1440, env_size: int=10):
        super(DecoderModel, self).__init__()
        self.d_model = d_model 
        self.nhead = nhead
        self.nb_actions = nb_requests + 1
        self.nb_vehicles = nb_vehicles
        self.nb_tokens =  nb_requests 
        self.nb_requests = nb_requests 
        self.device = get_device()
        self.time_end = time_end
        self.env_size = env_size

        self.encoder =  nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.encoders = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.nb_actions)
        self.decoder_linear = nn.Linear(self.d_model, self.nb_actions)
        self.PE = self.generate_positional_encoding(d_model, 1000).to(self.device)


        #Transformers
        self.request_encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.request_linear = nn.Linear(15 * d_model, d_model)

        #Request embeddings
        self.embed_time = nn.Embedding(time_end * 10 + 1, d_model) 
        self.embed_position = nn.Embedding(env_size * 2 * 10 + 1, d_model)
        self.embed_request_id = nn.Embedding(nb_requests + 1, d_model)
        self.embed_request_status = nn.Embedding(3, d_model)
        self.embed_vehicle_status = nn.Embedding(3, d_model)
        self.embed_current_vehicle = nn.Embedding(nb_vehicles, d_model)
        self.embed_being_served = nn.Embedding(nb_vehicles + 1, d_model)

    def embeddings(self, world, requests, vehicles):
        r = []
        world = world.float()
        requests = requests.float()
        vehicles = vehicles.float()
        world = torch.transpose(world, 0, 1) #world.size() = (8, bsz)
        vehicles = torch.transpose(vehicles, 0, 1) #vehicles.size() = (6, bsz)
        zeros = torch.zeros(world[1].size())
        result = []
        #requests.size() = (bsz, nb_requests, 10)
        for i in range(self.nb_requests):
            #position of current vehicle
            r = []
            r.append(self.embed_position(torch.round(vehicles[0] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(vehicles[1] * 10 + self.env_size * 10).long().to(self.device)))

            #max ride time and max route duration
            r.append(self.embed_time(torch.round(10 * world[6]).long().to(self.device)))
            #TODO: sometimes it is cropped at 1440 instead of being 480 (see: generator.py maybe?)

            r.append(self.embed_time(torch.round(10 * world[7]).long().to(self.device))) 
            r.append(self.embed_request_id(requests[:,i,0].long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,1] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,2] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,3] * 10 + self.env_size * 10).long().to(self.device)))
            r.append(self.embed_position(torch.round(requests[:,i,4] * 10 + self.env_size * 10).long().to(self.device)))
            shifted5 = torch.maximum(requests[:,i,5] - world[1], zeros)
            shifted6 = torch.maximum(requests[:,i,6] - world[1], zeros)
            shifted7 = torch.maximum(requests[:,i,7] - world[1], zeros)
            shifted8 = torch.maximum(requests[:,i,8] - world[1], zeros)
            r.append(self.embed_time(torch.round(shifted5 * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(shifted6 * 10).long().to(self.device))) 
            r.append(self.embed_time(torch.round(shifted7 * 10).long().to(self.device)))
            r.append(self.embed_time(torch.round(shifted8 * 10).long().to(self.device)))
            r.append(self.embed_request_status(requests[:,i,9].long().to(self.device)))
            r.append(self.embed_being_served(requests[:,i,10].long().to(self.device)))
            r = torch.stack(r).transpose(0, 1) #r.size() = (bsz, 15, d_model)
            r = self.request_encoder(r) 
            r = self.request_linear(r.flatten(start_dim=1))
            result.append(r)
        result = torch.stack(result).transpose(0, 1)
        return result
    
    def generate_positional_encoding(self, d_model, max_len):
        """
        Create standard transformer PEs.
        Inputs :  
        d_model is a scalar correspoding to the hidden dimension
        max_len is the maximum length of the sequence
        Output :  
        pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, state, deterministic=False):
        world, requests, vehicles = state 
        bsz = world.shape[0]
        zero_to_bsz = torch.arange(bsz, device=self.device)
        tours = []
        mask_visited_nodes = torch.zeros(self.nb_actions)
        one = torch.tensor(1).to(self.device) 
        r = self.embeddings(world, requests, vehicles)
        r = r.permute([1,0,2]) #r.size() = (nb_tokens, bsz, d_model)
        encoded = self.encoders(r)
        sum_log_prob_of_actions = []
        idx_start = 0
        h_t = encoded[0, :, :] + self.PE[0]
        t = 0
        for _ in range(self.nb_vehicles):
            tour = []
            #while action <= torch.tensor([self.nb_requests]):
            for i in range(3):
                t += 1
                h_t = h_t.unsqueeze(0)
                prob_next_node =   self.decoder_linear(self.decoder(h_t, encoded).flatten(start_dim=1))
                print("prob", prob_next_node.size())

                if deterministic:
                    idx = torch.argmax(prob_next_node, dim=1)
                else:
                    idx = Categorical(prob_next_node).sample() 
                
                # compute logprobs
                print("idx",idx.size())
                prob_of_choices = prob_next_node[idx]
                sum_log_prob_of_actions.append(torch.log(prob_of_choices))

                # update embeddings
                h_t = encoded[idx, :, :]
                h_t = h_t + self.PE[t+1]

                # update
                action = idx.clone()
                tours.append(action.cpu())

                # update masks with visited nodes
                #mask_visited_nodes[idx] += 1
            tours.append(tour)

        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sum_log_prob_of_actions = sum(sum_log_prob_of_actions) 

        return tours, sum_log_prob_of_actions



class Aoyu(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, nb_requests: int=16, nb_vehicles: int=2, num_layers: int=2, time_end: int=1440, env_size: int=10):
        super(Aoyu, self).__init__()
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

        self.encoder =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoders = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.classifier = nn.Linear(self.nb_tokens * d_model, self.nb_actions)

    def __repr__(self):
        return f"aoyu_{self.d_model}"

    def __str__(self):
        return f"aoyu_{self.d_model}"

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
            #TODO: sometimes it is cropped at 1440 instead of being 480 (see: generator.py maybe?)

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

    def act(self, state, mask):
        out = self.forward(state).cpu()
        out = out * mask
        mask = mask.type(torch.BoolTensor)
        out = torch.where(mask, out, torch.tensor(-1e+8).double())
        probs = nn.Softmax(dim=1)(out)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def greedy(self, state, mask):
        out = self.forward(state).cpu()
        out = out * mask
        mask = mask.type(torch.BoolTensor)
        out = torch.where(mask, out, torch.tensor(-1e+8).double())
        probs = nn.Softmax(dim=1)(out)
        action = probs.max(1)[1].view(1, 1)
        return action.item(), 0


class Policy(nn.Module):
    def __init__(self, d_model: int=512, nhead: int=8, nb_requests: int=16, nb_vehicles: int=2, num_layers: int=2, time_end: int=1440, env_size: int=10):
        super(Policy, self).__init__()
        self.nb_actions = nb_requests + 1
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
        self.embed_vehicle_status = nn.Embedding(3, d_model)
        self.embed_current_vehicle = nn.Embedding(nb_vehicles, d_model)

    def embeddings(self, world, requests, vehicles):
        world = world.float()
        requests = requests.float()
        vehicles = vehicles.float()
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
        action = probs.max(1)[1].view(1, 1)
        return action.item(), 0


def reinforce(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int, 
             update_baseline: int,
             envs: List[DarpEnv],
             test_env: DarpEnv):
    #init baseline model
    baseline = copy.deepcopy(policy)
    baseline = baseline.to(device)
    policy = policy.to(device)
    train_R = 0
    baseline_R = 0
    routes = []
    penalties = []
    train_losses = []
    for i_epoch in range(nb_epochs):
        logger.info("*** EPOCH %s ***", i_epoch)
        for i_episode, env in enumerate(envs):
            logger.info("episode: %s, RAM: %s, CPU: %s", i_episode, psutil.virtual_memory().percent, psutil.cpu_percent())
            max_step = env.nb_requests * 2 + env.nb_vehicles

            env.reset()
            #update baseline model after every n steps
            if i_episode % update_baseline == 0:
                if train_R <= baseline_R:
                    logger.info("new baseline model selected after achiving %s reward", train_R)
                    baseline.load_state_dict(policy.state_dict())

            #simulate episode with train and baseline policy
            with torch.no_grad():
                baseline_rewards, _ = evaluate.simulate(max_step, env, baseline, greedy=True)
                env.penalize_broken_time_windows()
                baseline_penalty = env.penalty["sum"]
                #TODO: this is wrong we cannot restore the same env insted give it as parameters [requests, vehicles, depots], 
                #TODO: we wil only need one env, generate new configuration after each episode
                env.reset()  
                
            rewards, log_probs = evaluate.simulate(max_step, env, policy)
            env.penalize_broken_time_windows()
            penalty = env.penalty["sum"]

            #aggregate rewards and logprobs
            train_R = sum(rewards) + penalty
            baseline_R = sum(baseline_rewards) + baseline_penalty
            sum_log_probs = sum(log_probs)

            policy_loss = torch.mean((train_R - baseline_R) * sum_log_probs)
        
            #backpropagate
            optimizer.zero_grad()
            policy_loss.backward()
            train_losses.append(policy_loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()

            #garbage collector
            del(env)
            gc.collect()

            #test pahse
            if i_episode % 100 == 0:
                with torch.no_grad():
                    test_env.reset()
                    route, penalty = evaluate.evaluate_model(policy, test_env, max_step, i=i_episode)
                    routes.append(route)
                    penalties.append(penalty)
            if i_episode > 1000:
                break
    
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


def reinforce_trainer(train_envs_path: str, 
                        test_env_path: str, 
                        result_path: str,
                        id: str,
                        nb_epochs: int, 
                        policy: Policy, 
                        optimizer: torch.optim.Optimizer):
                        
    train_envs = load_data(train_envs_path)
    test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=100, dataset=test_env_path)
    logger.info("dataset successfully loaded")

    logger.info("training starts")
    result = reinforce(policy, optimizer, nb_epochs, update_baseline=100, envs=train_envs,test_env=test_env)
    torch.save(policy.state_dict(), result_path + '/' + id + "-model")
    return result


if __name__ == "__main__":
    seed_everything(1)
    train_envs_path = "data/processed/generated-10000-a2-16.pkl"
    instance = "a2-16"
    #train_envs, test_envs = load_aoyo(instance)
    test_env_path = f'data/cordeau/{instance}.txt'  
    #env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=1000, dataset=FILE_NAME)

    result_path = "models"
    nb_epochs = 1

    policy = Aoyu(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    device = get_device()
    #PATH = "models/result-a4-48-supervised-rf-01-aoyu-model"
    #PATH = "models/result-a2-16-supervised-rf-01-aoyu-model"
    #state = torch.load(PATH)
    #policy.load_state_dict(state)
    logger.info("training on device: %s", device)
    policy = policy.to(device)
    logger.info(policy)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-3)

    logger = set_level(logger, "info")
    id = "result-a2-16-reinforce-01-aoyu"
    reinforce_trainer(train_envs_path, test_env_path, result_path, id ,nb_epochs, policy, optimizer)
  
    


