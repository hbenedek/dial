from typing import List, Tuple
import pickle
import urllib
import requests
import numpy as np
from bs4 import BeautifulSoup
import re
import sys
from env import DarpEnv, Request, Vehicle

def tabu_scraper(path = "..data/tabu/"):
    """scrapes the benchmark dataset from Cordeau, Laporte 2003"""
    for i in range(1,21):
        instance = str(i).zfill(2)
        url = f"http://neumann.hec.ca/chairedistributique/data/darp/tabu/pr{instance}"
        filename = f"pr{instance}.txt"
        urllib.request.urlretrieve(url, path+filename)

def branch_cut_scraper(path = "..data/cordeau/"):
    """scrapes the benchmark dataset from Cordeau 2006"""
    root = "http://neumann.hec.ca/chairedistributique/data/darp/branch-and-cut/"
    page = requests.get(root).text
    soup = BeautifulSoup(page, 'html.parser')
    urls = [root + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('')][1:]
    
    for url in urls:
        filename = re.sub("(.*)//","", url)
        url = root + filename
        print(f"scraping {url}")
        urllib.request.urlretrieve(url, path+filename+".txt")


def generate_training_data(
                        N: int, #number of instances to generate
                        size: int, 
                        nb_vehicles: int,
                        nb_requests: int,
                        time_end: int,
                        max_step: int,
                        max_route_duration: int,
                        capacity: int,
                        max_ride_time: int,
                        window: bool
                        ) -> List[DarpEnv]: 
    envs = []    
    for _ in range(N):
        env = DarpEnv(size, nb_requests, nb_vehicles, time_end, max_step, max_route_duration, capacity, max_ride_time, window=window)
        envs.append(env)
    return envs

def dump_training_data(envs: List[DarpEnv], file: str):
    with open(file, 'wb') as handle:
        pickle.dump(envs , handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_training_data(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


if __name__ == "__main__":
        
    envs = generate_training_data(N=1000,
                        size= 10, 
                        nb_vehicles=2,
                        nb_requests=16,
                        time_end=1400,
                        max_step=1000,
                        max_route_duration=480,
                        capacity=3,
                        max_ride_time=30,
                        window=True)

    path = "../data/test_sets/generated-a2-16.pkl"
    dump_training_data(envs, path)

if __name__ == "__main__":
    print(sys.path)
