from model import Policy
import torch
from typing import List, Tuple, Optional
from env import DarpEnv
import torch.nn as nn
import numpy as np
from log import logger
from torch.utils.data import DataLoader
from generator import load_data

def generate_supervised_dataset(max_step: int, envs: List[DarpEnv], test_size: float, batch_size: int, size: Optional[int]=None) -> Tuple[DataLoader, DataLoader]:
    dataset = []
    i = 0
    for env in envs:
        obs = env.reset()
        for t in range(max_step):
            i += 1
            action = env.nearest_action_choice()
            if action == -1:
                print("WARNING")
            dataset.append([obs, action])
            if size:
                if i > size:
                    break
            obs, _, done = env.step(action)
            if done:
                break

    split_idx = int(len(dataset) * (1 - test_size))
    train, test = dataset[:split_idx], dataset[split_idx:]
    logger.info("Supervised train dataset generetad of size %s", len(train))
    logger.info("Supervised train dataset generetad of size %s", len(test))
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def copycat_trainer(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int,
             train_loader,
             test_loader):
    """Trains the Transformer policy network against the nearest neighbour policy"""
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(outputs, supervised_actions)

            loss.backward()
            optimizer.step()
            total += supervised_actions.size(0)
            model_actions = torch.max(outputs, 1).indices
            correct += np.sum((model_actions == supervised_actions).cpu().numpy())
            running_loss += loss.item()

        acc = 100 * correct/ total
        logger.info("EPOCH %s: train_loss: %s train_acc: %s", epoch, running_loss, acc)

        #test phase
        policy.eval()
        running_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader):
            states, supervised_actions = data
            with torch.no_grad():
                outputs= policy(states)

            loss = criterion(outputs, supervised_actions)
            total += supervised_actions.size(0)
            model_actions = torch.max(outputs, 1).indices
            correct += np.sum((model_actions == supervised_actions).cpu().numpy())
            running_loss += loss.item()
        acc = 100 * correct/ total
        logger.info("EPOCH %s: test_loss: %s test_acc: %s", epoch, running_loss, acc)

    return 0


if __name__ == "__main__":

    envs = load_data("data/test_sets/generated-a2-16.pkl")
    train_loader, test_loader = generate_supervised_dataset(max_step=100, envs=envs, test_size=0.2, batch_size=16, size=None)

    policy = Policy(d_model=100, nhead=4, nb_requests=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    copycat_trainer(policy=policy, optimizer=optimizer, nb_epochs=1, train_loader=train_loader, test_loader=test_loader)
    #TODO: add result class

    
