import copy
import torch.nn as nn
import torch
from torch.distributions import Categorical

class MonteCarloTreeSearch():
    pass


class BeamSearch():
    def __init__(self, policy, env, k):
        self.policy = policy
        self.k = k
        self.k_best = [[list(), env, 0.0, False]] #{sequence, env, score}

    def act(self, env):
        if sum([tup[3] for tup in self.k_best]) == self.k:
            return self.k_best
        #walk over each step in sequence
        candidates = []
        for s, env, score, done in self.k_best:
            if not done:
                world, requests, vehicle = env.representation()
                state = [torch.tensor(world).unsqueeze(0), torch.tensor(requests).unsqueeze(0), torch.tensor(vehicle).unsqueeze(0)]
                out = self.policy(state)
                probs = nn.Softmax(dim=1)(out)
                m = Categorical(probs)
                log_probs, actions = torch.topk(m.log_prob, self.k)
                for log_prob, action in zip(log_probs, actions):
                    #expand each current candidate
                    candidates.append([s + [action], copy.deepcopy(env), score + log_prob])

        #order all candidates by score
        ordered = sorted(candidates, key=lambda tup:tup[2])
        state, reward, done = env.step(action)
        #select k best
        ordered = ordered[:self.k]
        self.k_best = []
        #step in every potential env
        for actions, env, scores in ordered:
            state, reward, done = env.step(action)
            self.k_best.append([actions, env, scores, done])


        
        


