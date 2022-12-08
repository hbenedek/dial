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

    df1 = pd.read_csv("evaluations/new-data-a2-16-test-model-nn-a2-16")
    df1["model"] = "nn-a2-16"
    df2 = pd.read_csv("evaluations/new-data-a2-16-test-model-rf-a2-16")
    df2["model"] = "rf-a2-16"
    df3 = pd.read_csv("evaluations/new-data-a2-16-test-model-rf-a4-48")
    df3["model"] = "rf-a4-48"
    df = pd.concat([df1, df2, df3]).drop("Unnamed: 0", axis=1)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    fig.suptitle("data: a2-16")
    
    ax[0].set(title="gap distribution", xlabel="gap", ylabel="counts")

    sns.histplot(data=df, x="gap", hue="model", ax=ax[0], fill=True, alpha=0.5, element="step")
    #labels = ax[0].get_yticks().tolist()
    #labels = [float(label) * 10000 for label in labels]
    #ax[0].set_yticklabels(labels)

    ax[1].set(title="penalty distribution", xlabel="penalty", ylabel="counts")
    sns.histplot(data=df, x="penalty", hue="model", ax=ax[1], fill=True, alpha=0.5, element="step")

    #labels = ax[1].get_yticks().tolist()
    #labels = [round(float(label) * 10000 * 23,2) for label in labels]
    #ax[1].set_yticklabels(labels)

    print(df.groupby("model").mean())

    # PATH = "models/result-a2-16-supervised-rf-05-aoy4layer"
    # r1 = load_data(PATH)
    # PATH = "models/result-a2-16-supervised-rf-06-aoy4layer"
    # r11 = load_data(PATH)

    # PATH = "models/result-a2-16-supervised-nn-06-aoy4layer"
    # r2 = load_data(PATH)

    # PATH = "models/result-a4-48-supervised-rf-06-aoy4layer"
    # r3 = load_data(PATH)
    
    # fig, ax = plt.subplots(1,2, figsize=(12,4))
    # fig.suptitle("Supervised policy stealing on 10.000 a2-16 instance")
    # epoch = [str(i) for i in range(1,21)]
    # ax[0].set(title="train loss", xlabel="epoch", ylabel="cross entropy")
    # ax[0].plot(epoch, r2.train_loss, label="nn-a2-16")
    # ax[0].plot(epoch, r1.train_loss + r11.train_loss, label="rf-a2-16")
    # ax[0].plot(epoch[:10], r3.train_loss, label="rf-a4-48")
    # for label in ax[0].xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)
    # ax[1].set(title="test accuracy", xlabel="epoch", ylabel="accuracy")
    # ax[1].plot(epoch, r2.accuracy, label="nn-a2-16")
    # ax[1].plot(epoch, r1.accuracy + r11.accuracy, label="rf-a2-16")
    # ax[1].plot(epoch[:10], r3.accuracy, label="rf-a4-48")
    # for label in ax[1].xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)

    # plt.legend()
    # plt.savefig("train.png")



