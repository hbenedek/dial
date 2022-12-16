import copy
import torch.nn as nn
import torch
from torch.distributions import Categorical
import numpy as np
from env import DarpEnv
from model import Policy
from utils import get_device
from generator import load_data
from collections import deque

class BeamSearch():
    def __init__(self, policy, k):
        self.policy = policy
        self.k = k

    def evaluate(self, env):
        mask_value = torch.tensor(-1e+8).double().to(self.policy.device)
        i = 0
        self.k_best = [(list(), env, 0.0, False, 0.0, list())] #{sequence, env, score, done, total_travel, indices}
        while sum([tup[3] for tup in self.k_best]) < self.k:
            i += 1
            #walk over each step in sequence
            candidates = []
            for seq, env, score, done, _, indices in self.k_best:
                if not done:
                    if not indices:
                        free_times = [vehicle.free_time for vehicle in env.vehicles]
                        env.current_time = np.min(free_times)
                        indices = np.argwhere(free_times == env.current_time)
                        indices = deque(indices.flatten().tolist())
                        print(indices, free_times)
                    
                    vehicle_id = indices.popleft()
                    if env.vehicles[vehicle_id].state != "finished":
                        world, requests, vehicle = env.representation()
                        state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
                        mask = env.mask_illegal(vehicle_id).to(self.policy.device)
                        print(vehicle_id, mask)
                        print("history", env.vehicles[vehicle_id].history)
                        out = self.policy(state)
                        out = out * mask
                        mask = mask.type(torch.BoolTensor).to(self.policy.device)
                        out = torch.where(mask, out, mask_value)
                        probs = nn.Softmax(dim=1)(out)

                    log_probs, actions = torch.topk(torch.log(probs.squeeze(0)), self.k)

                    for log_prob, action in zip(log_probs, actions):
                         #expand each current candidate
                        candidates.append([seq + [action.item()], copy.deepcopy(env), score - log_prob.item(), indices])
            #order all candidates by score
            ordered = sorted(candidates, key=lambda tup:tup[2])
            #select k best
            ordered = ordered[:self.k]
            self.k_best = []
            #step in every potential env
            for i, (actions, env, scores, indices) in enumerate(ordered):
                action = actions[-1]
                state, reward, done = env.step(action, vehicle_id)
                total = sum([v.total_distance_travelled for v in env.vehicles])
                self.k_best.append([actions, env, scores, done, total, indices])
                
        return self.k_best



if __name__ == "__main__":
    import re
    instance = "a2-16"
    FILE_NAME = f'data/cordeau/{instance}.txt'
    nb_requests = int(re.search(".*-(.*)", instance).group(1))
    nb_vehicles = int(re.search("[ab](.*)-.*", instance).group(1))
        
    test_env = DarpEnv(size=10, nb_requests=nb_requests, nb_vehicles=nb_vehicles, time_end=1440, dataset=FILE_NAME)
    policy = Policy(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
    path = "models/new-resulta2-16-supervised-rf"
    r = load_data(path)
    state = r.policy_dict
    policy.load_state_dict(state)
    device = get_device()
    policy = policy.to(device)
    policy.eval()
    beam = BeamSearch(policy, 1)
    print("evaluation starts")
    bestk = beam.evaluate(test_env)
    for b in bestk:
        print(b[4])


        
        


