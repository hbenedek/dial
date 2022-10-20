import numpy as np
import logging
import torch
import urllib
import requests
from bs4 import BeautifulSoup
import re

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

def float_equality(f1: float, f2: float, eps: float=0.001) -> bool:
    return abs(f1 - f2) < eps

def distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.linalg.norm(pos1 - pos2)

def init_logger(log_path: str="../logs", file_name: str="log", level: str = "info") -> logging.Logger:
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    if level == "info":
        logger.setLevel(logging.INFO)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    return logger

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


if __name__ == "__main__":
    tabu_scraper()
    branch_cut_scraper()