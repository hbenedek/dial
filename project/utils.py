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
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from generator import load_data

    df = pd.read_csv("evaluations/new-data-b3-24-baseline")
    df["model"] = "nn-baseline"
    #df2 = pd.read_csv("evaluations/new-data-a2-16-test-model-rf-a2-16")
    #df2["model"] = "rf-a2-16-20-epochs"
    #df3 = pd.read_csv("evaluations/new-data-a2-16-test-model-rf-a4-48")
    #df3["model"] = "rf-a4-48"
    #df = pd.concat([df1, df2, df3]).drop("Unnamed: 0", axis=1)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    fig.suptitle("data: a2-16")
    
    ax[0].set(title="gap distribution", xlabel="gap", ylabel="counts")
    sns.histplot(data=df, x="gap", hue="model", ax=ax[0], fill=True, alpha=0.5, element="step")
   

    ax[1].set(title="penalty distribution", xlabel="penalty", ylabel="counts")
    sns.histplot(data=df, x="penalty", hue="model", ax=ax[1], fill=True, alpha=0.5, element="step")
    plt.savefig("nnbaseline.png")
    #print(df.groupby("model").mean())

    
    # PATH = "models/result-a4-48-supervised-rf-06-aoy4layer"
    # r3 = load_data(PATH)

    
    # fig, ax = plt.subplots(1,2, figsize=(12,4))
    # fig.suptitle("Supervised policy stealing on b2-16 (train on 10.000, test on 1.000 instances)")
    # epoch = [str(i) for i in range(1,51)]
    # ax[0].set(title="Loss", xlabel="epoch", ylabel="cross entropy")
    # ax[0].plot(epoch, r2.train_loss, label="train_loss")
    # ax[0].plot(epoch, r2.test_loss, label="test_loss")
    # ax[0].plot(epoch, r1.train_loss + r11.train_loss, label="rf-a2-16")
    # ax[0].plot(epoch[:10], r3.train_loss, label="rf-a4-48")
    # for i, label in enumerate(ax[0].xaxis.get_ticklabels()[::1]):
    #     if not i % 5 == 4:
    #      label.set_visible(False)
    # ax[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    # ax[1].plot(epoch, train_acc, label="train_accuracy")
    # ax[1].plot(epoch, r2.accuracy, label="test_accuracy")
    # ax[1].plot(epoch, r1.accuracy + r11.accuracy, label="rf-a2-16")
    # ax[1].plot(epoch[:10], r3.accuracy, label="rf-a4-48")
    # for i, label in enumerate(ax[1].xaxis.get_ticklabels()[::1]):
    #     if not i % 5 == 4:
    #      label.set_visible(False)

    # ax[1].legend()
    # ax[0].legend()
    # plt.savefig("50btrain.png")

