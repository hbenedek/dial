import numpy as np
import logging
import torch
import re
import random
import os
import matplotlib.pyplot as plt


def float_equality(f1: float, f2: float, eps: float=0.001) -> bool:
    return abs(f1 - f2) < eps

def distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.linalg.norm(pos1 - pos2)

def coord2int(coord: float) -> int:
    """given a float transforms it to integer representation"""
    number_precision = 3
    new_coord = int(round(coord, number_precision)*(10**number_precision))
    return new_coord

def get_device() -> str:
    if torch.cuda.is_available(): 
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def plot_env(env):    
    #fig, ax = plt.subplots(1, 1, figsize=(10,10))
    x = np.stack([r.pickup_position for r in env.requests])
    print(x)
    y = np.stack([r.dropoff_position for r in env.requests])

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ab_pairs = np.c_[x, y]
    ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)

    # segments
    ax[0].plot(0, 0, 'gs', label="depot", markersize=13)
    ax[0].plot(*ab_args, c='k')
    ax[0].plot(*x.T, 'bo', label="pickup",markersize=13)
    ax[0].plot(*y.T, 'ro', label="dropoff",markersize=13)
    for i, r in enumerate(env.requests):
        ax[0].annotate(i, (r.pickup_position[0], r.pickup_position[1] + 0.2),size=12) 
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[0].get_yticklabels(), visible=False)
    ax[0].legend()
    ax[0].set(xlabel="x coordinate", ylabel="y coordinate", title="Darp env")
    
    ax[1].set(ylabel="Requests", xlabel="time", title="Pickup and Dropoff time window")
    ax[1].set_yticks(np.arange(16))
    for i, r in enumerate(env.requests):
        ax[1].broken_barh([(r.start_window[0], r.start_window[1] - r.start_window[0])], (i - 0.25, 0.5), facecolors='tab:blue')
        ax[1].broken_barh([(r.end_window[0], r.end_window[1] - r.end_window[0])], (i + 0.25, 0.5), facecolors='tab:red')

    plt.show()

if __name__ == "__main__":
    pass

