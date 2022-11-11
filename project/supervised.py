from model import Policy
import torch
from typing import List, Tuple, Optional
from env import DarpEnv
import torch.nn as nn
import numpy as np
from log import logger
from torch.utils.data import DataLoader
from entity import Result
from generator import dump_data, load_data
from utils import get_device


def generate_supervised_dataset(max_step: int, envs: List[DarpEnv], test_size: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset = []
    for env in envs:
        obs = env.reset()
        for _ in range(max_step):
            action = env.nearest_action_choice()
            dataset.append([obs, action])
            obs, _, done = env.step(action)
            if done:
                break

    split_idx = int(len(dataset) * (1 - test_size))
    train, test = dataset[:split_idx], dataset[split_idx:]
    logger.info("Supervised train dataset generetad of size %s", len(train))
    logger.info("Supervised test dataset generetad of size %s", len(test))
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    #TODO: save and dump loader objects
    return train_loader, test_loader


def copycat_trainer(policy: Policy,
             optimizer: torch.optim.Optimizer,
             nb_epochs: int,
             train_loader: DataLoader,
             test_loader: DataLoader,
             id: str) -> Result:
    """
    Trains the Transformer policy network against other policies depending on the dataloaders 
    (for the moment the nearest neighbour policy)
    """
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    accuracies = []
    for epoch in range(nb_epochs): 
        running_loss = 0
        total = 0
        correct = 0
        policy.train()
        #train phase
        for i, data in enumerate(train_loader):
            states, supervised_actions = data
            supervised_actions = supervised_actions.to(policy.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = policy(states)

            loss = criterion(outputs, supervised_actions)

            loss.backward()
            optimizer.step()

            total += supervised_actions.size(0)
            model_actions = torch.max(outputs, 1).indices
            correct += np.sum((model_actions == supervised_actions).cpu().numpy())
            running_loss += loss.item()
    
        acc = 100 * correct/ total
        train_losses.append(running_loss)
        logger.info("EPOCH %s: train_loss: %s train_acc: %s", epoch, round(running_loss, 4), round(acc, 4))

        #test phase
        policy.eval()
        running_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader):
            states, supervised_actions = data
            supervised_actions = supervised_actions.to(policy.device)
            with torch.no_grad():
                outputs= policy(states)

            loss = criterion(outputs, supervised_actions)
            total += supervised_actions.size(0)
            model_actions = torch.max(outputs, 1).indices
            correct += np.sum((model_actions == supervised_actions).cpu().numpy())
            running_loss += loss.item()
        acc = 100 * correct/ total
        test_losses.append(running_loss)
        accuracies.append(acc)
        logger.info("EPOCH %s: test_loss: %s test_acc: %s", epoch, round(running_loss, 4), round(acc, 4))

    # save results in Result object
    result = Result(id)
    result.train_loss = train_losses
    result.test_loss = test_losses
    result.accuracy= accuracies
    result.policy_dict = policy.state_dict()
    return result

def supervised_trainer(envs_path: str, 
                        result_path: str,
                        max_steps: int,
                        test_size: float,
                        batch_size: int,
                        nb_epochs: int,
                        policy: Policy,
                        optimizer:torch.optim.Optimizer,
                        id: str) -> Result:

    envs = load_data(envs_path)
    logger.info("dataset successfully loaded")

    train_loader, test_loader = generate_supervised_dataset(max_step=max_steps, envs=envs, test_size=test_size, batch_size=batch_size)
    logger.info("train and test DataLoader objects successfully initialized")

    logger.info("training starts")
    result = copycat_trainer(policy=policy, optimizer=optimizer, nb_epochs=nb_epochs, train_loader=train_loader, test_loader=test_loader, id=id)

    logger.info("saving Result object...")
    dump_data(result, result_path + '/' + id)
    logger.info("saving done")
    return result


if __name__ == "__main__":
    envs_path = "data/processed/generated-10000-a2-16.pkl"
    result_path = "models"
    max_steps = 1440 * 16
    test_size = 0.05
    batch_size = 128
    nb_epochs = 10
    policy = Policy(d_model=128, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=2, time_end=1440, env_size=10)
    device = get_device()
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-3)
    id = "result-a2-16-supervised-nn-05"
    #envs = load_data(envs_path)
    #logger.info("dataset successfully loaded")

    #train_loader, test_loader = generate_supervised_dataset(max_step=max_steps, envs=envs, test_size=test_size, batch_size=batch_size)
    #logger.info("train and test DataLoader objects successfully initialized")
    #for i, data in enumerate(train_loader):
    #    states, supervised_actions = data
    #    world, requests, vehicles = states 
    #  
    #    out = policy(states)
    #    logger.info("loaded %s", i)
    result = supervised_trainer(envs_path, result_path, max_steps, test_size, batch_size, nb_epochs, policy, optimizer, id) 
    






    
