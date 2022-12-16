from model import Policy
import torch
from typing import List, Tuple, Optional
from env import DarpEnv
import torch.nn as nn
import numpy as np
from log import logger, set_level
from torch.utils.data import DataLoader
from entity import Result
from generator import dump_data, load_data, load_aoyo
from evaluate import evaluate_aoyu
from utils import get_device
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.99)

    running_loss = 0
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
        train_losses.append(running_loss / (i + 1))
        logger.info("EPOCH %s: train_loss: %s train_acc: %s", epoch, round(running_loss / (i + 1), 4), round(acc, 4))

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
        test_losses.append(running_loss / (i + 1))
        accuracies.append(acc)
        logger.info("EPOCH %s: test_loss: %s test_acc: %s", epoch, round(running_loss / (i + 1), 4), round(acc, 4))

    # save results in Result object
    policy = policy.to("cpu")
    state_dict = policy.state_dict()

    result = Result(id)
    result.train_loss = train_losses
    result.test_loss = test_losses
    result.accuracy= accuracies
    result.policy_dict = state_dict
    return result

def extract_state_action_pairs(envs, act, batch_size, augment=[]):
    dataset = []
    for env in envs:
        try:
            nb_requests, nb_vehicles = env.nb_requests, env.nb_vehicles
            if augment:
                nb_vehicles_augmented, nb_requests_augmented = augment
                logger.debug("padding env")
                env.pad_env(nb_vehicles_augmented, nb_requests_augmented) 
                logger.debug("augmenting env")
                # shuffle Request actions, leave return to end depot action as last 
                permutation = np.random.permutation(nb_requests_augmented) #np.arange(nb_requests_augmented)  
                env.augment(permutation) #TODO: somehow not working
                permutation = list(np.append(permutation, nb_requests_augmented))
            obs = env.representation()
            for _ in range(2 * nb_requests + nb_vehicles):
                action = act(env)
                if augment:
                    action = permutation.index(action)
                dataset.append([obs, action])
                obs, _, done = env.step(action)
                if done:
                    break
        except:
            logger.info("PROBLEM OCCURED DURING EXTRACTING STATE ACTION PARIS")
            logger.info(permutation)
    logger.info("Supervised dataset generetad of size %s", len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def generate_supervised_dataset(train_envs, test_envs, supervised_policy: str, batch_size: int, augment=[]):
    if supervised_policy == "nn":
        def act(env): 
            return env.nearest_action_choice()
    if supervised_policy == "rf":
        def act(env): 
            action = env.vehicles[env.current_vehicle].routes.popleft() - 1 
            if action == env.original_nb_requests * 2:
                return env.nb_requests
            if action >= env.original_nb_requests:
                action = action - env.original_nb_requests
            return action
    train_loader = extract_state_action_pairs(train_envs, act, batch_size, augment)
    test_loader = extract_state_action_pairs(test_envs, act, batch_size, augment)
    return train_loader, test_loader


def supervised_trainer( 
                    id: str,
                    instance: str,
                    result_path: str,
                    supervised_policy: str,
                    batch_size: int,
                    nb_epochs: int,
                    policy: Policy,
                    optimizer:torch.optim.Optimizer,
                    augment: List[int]=[],
                    env_path: str="data/test_sets/generated-10000-a2-16.pkl") -> Result:

    if instance:
        train_envs, test_envs = load_aoyo(instance)
    else:
        envs = load_data(env_path)
        split = len(envs) * 0.98
        train_envs, test_envs = envs[:split], envs[split:]

    logger.info("training id: %s", id)

    train_loader, test_loader = generate_supervised_dataset(train_envs, test_envs, supervised_policy, batch_size, augment)
    logger.info("train and test DataLoader objects successfully initialized")

    logger.info("training starts")
    result = copycat_trainer(policy=policy, optimizer=optimizer, nb_epochs=nb_epochs, train_loader=train_loader, test_loader=test_loader, id=id)
    result.instance = instance
    logger.info("saving Result object...")
    dump_data(result, result_path + '/' + id)
    logger.info("saving done")
    return result


if __name__ == "__main__":

    ################################    EXAMPLE USAGE 1 (SUPERVISED TRAINING)   #######################################
    import re
    import glob
    result_path = "models/"
    batch_size = 256
    nb_epochs = 20
    supervised_policy="rf"

    #logger = set_level(logger, "debug")
    #initialize policy
    policy = Policy(d_model=64, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=8, time_end=1440, env_size=10)
    #path = "models/new-result-a4-48-supervised-rf"
    #r = load_data(path)
    #state = r.policy_dict
    #policy.load_state_dict(state)

    # #initialize optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    #id = "result-02-16-supervised-rf-50-epoch"
    files = glob.glob("data/aoyu/[ab]*")
    instances = set([re.search("data/aoyu/(.*-.*)-.*", file).group(1) for file in files])
    #start train
    instances=["a2-16"]
    for i, instance in enumerate(instances):
        id = f"augmented-{instance}"
        device = get_device()
        policy = policy.to(device)
        logger.info("ITERATION %s, training on device: %s", i, device)
        result = supervised_trainer(id, 
                                instance,
                                result_path,
                                supervised_policy,
                                batch_size, 
                                nb_epochs, 
                                policy,
                                optimizer)#,augment=[4, 48]) 

    

    

   






    
