from model import Policy
import torch
from typing import List, Tuple
from env import DarpEnv
import torch.nn as nn
import numpy as np
from log import logger
from torch.utils.data import DataLoader

def generate_supervised_dataset(max_step: int, envs: List[DarpEnv], test_size: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset = []
    for env in envs:
        obs = env.reset()
        for t in range(max_step):
            action = env.nearest_action_choice()
            dataset.append([obs, action])

            obs, _, done = env.step(action)
            if done:
                break

    split_idx = int(len(dataset) * (1 - test_size))
    train, test = dataset[:split_idx], dataset[split_idx:]
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def copycat_trainer(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int,
             max_step: int, 
             train_loader,
             test_loader):

    criterion=nn.MSELoss() 
    for epoch in range(nb_epochs): 
        running_loss = 0
        total = 0
        correct = 0
        policy.train()
        #train phase
        for i, data in enumerate(train_loader):
            states, supervised_actions = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs= policy(states)
            model_action = outputs.max(1)[1].view(1, 1)
            loss = criterion(outputs, supervised_actions)

            loss.backward()
            optimizer.step()

            total += supervised_actions.size(0)
            correct += np.sum((model_action.squeeze(1).argmax(-1) == supervised_actions.squeeze(-1)).cpu().numpy())
            running_loss += loss.item()

        acc = 100 * correct/ total
        #TODO: log train results
        #test phase
        policy.eval()
        running_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader):
            states, supervised_actions = data
            with torch.no_grad():
                outputs= policy(states)

            model_action = outputs.max(1)[1].view(1, 1)
            loss = criterion(outputs, supervised_actions)
            total += supervised_actions.size(0)
            correct += np.sum((model_action.squeeze(1).argmax(-1) == supervised_actions.squeeze(-1)).cpu().numpy())
            running_loss += loss.item()

        #TODO: log test results

    pass

