from typing import List, Tuple
import pickle
import urllib
import requests
import numpy as np
from bs4 import BeautifulSoup
import re
import sys
from entity import Request, Vehicle

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


def generate_instance(seed,
                    size, 
                    nb_vehicles, 
                    nb_requests, 
                    time_end, 
                    capacity, 
                    max_route_duration, 
                    max_ride_time,
                    window=None) -> Tuple[List[Vehicle], List[Request], List[np.ndarray]]:
        """"generates random pickup, dropoff and other constraints, a list parsed of Vehicle and Request objects"""
        if seed:
            np.random.seed(seed)

        target_pickup_coodrs = np.random.uniform(-size, size, (nb_requests, 2))
        target_dropoff_coords = np.random.uniform(-size, size, (nb_requests, 2))

        #generate depot coordinates
        start_depot = np.random.uniform(-size, size, 2)
        end_depot = np.random.uniform(-size, size, 2)
        depots = [start_depot, end_depot]

        #generate time window constraints
        if window:
            start_windows, end_windows  = generate_window(nb_requests, time_end, max_ride_time)
        else:
            start_windows = [np.array([0, time_end]) for _ in range(nb_requests)]
            end_windows = [np.array([0, time_end]) for _ in range(nb_requests)]

        #init Driver and Target instances
        vehicles = []
        for i in range(nb_vehicles):
            driver = Vehicle(id=i,
                            position=start_depot,
                            capacity=capacity,
                            max_route_duration=max_route_duration)
            vehicles.append(driver)

        requests = []
        for i in range(nb_requests):
            request = Request(id=i,
                            pickup_position=target_pickup_coodrs[i],
                            dropoff_position=target_dropoff_coords[i],
                            #represents the earliest and latest time, which the service may begin
                            start_window=start_windows[i],
                            end_window=end_windows[i],
                            max_ride_time=max_ride_time)
            requests.append(request)
        return vehicles, requests, depots


def parse_data(datadir: str) -> Tuple[List[Vehicle], List[Request], List[np.ndarray]]:
        """given a cordeau2006 instance, the function returns a list parsed of Vehicle and Request objects"""
        file_name = datadir
        with open(file_name, 'r') as file :
            number_line = sum(1 if line and line.strip() else 0 for line in file if line.rstrip()) - 3
            file.close()

        with open(file_name, 'r') as file :
            nb_vehicles, nb_requests, max_route_duration, capacity, max_ride_time = list(map(int, file.readline().split()))

            #Depot
            _, depo_x, depo_y, _, _, _, _ = list(map(float, file.readline().split()))
            start_depot = np.array([depo_x, depo_y])
            end_depot = np.array([depo_x, depo_y])
            depots = [start_depot, end_depot]

            #Init vehicles
            vehicles = []
            for i in range(nb_vehicles):
                vehicle = Vehicle(id=i,
                            position=start_depot,
                            capacity=capacity,
                            max_route_duration=max_route_duration)
                vehicles.append(vehicle)

            #Init requests
            requests = []
            for l in range(number_line):
                #parsing line 1, ...,n
                if l < number_line // 2:
                    identity, pickup_x, pickup_y, _, _, start_tw, end_tw = list(map(float, file.readline().split()))

                    request = Request(id=identity,
                                pickup_position=np.array([pickup_x, pickup_y]),
                                dropoff_position=None,
                                #represents the earliest and latest time, which the service may begin
                                start_window=np.array([start_tw, end_tw]),
                                end_window=None,
                                max_ride_time=max_ride_time)

                    requests.append(request)
                #parsing line n+1, ..., 2n
                else:
                    _, dropoff_x, dropoff_y, _, _, start_tw, end_tw = list(map(float, file.readline().split()))
                    request = requests[l - number_line // 2]
                    request.dropoff_position = np.array([dropoff_x, dropoff_y])
                    request.end_window = np.array([start_tw, end_tw])

        return vehicles, requests, depots

def generate_window(nb_requests: int, time_end: int, max_ride_time: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        #FIXME: this method does not seem very sophisticated, copied from Pawal's code
        """Generate 50% of free dropof conditions, and 50% of free pickup time conditions time windows for Requests"""
        start_windows = []
        end_windows = []
        for j in range(nb_requests):
            # Generate start and end point for window
            start = np.random.randint(0, time_end * 0.9) 
            end = np.random.randint(start + 15, start + 45)
            #free pickup condition
            if j < nb_requests // 2:
                start_fork = [max(0, start - max_ride_time), end]
                end_fork = [start, end]
            #free dropoff condition
            else:
                start_fork = [start, end]
                end_fork = [start, min(time_end, end + max_ride_time)]

            start_windows.append(np.array(start_fork))
            end_windows.append(np.array(end_fork))
        return start_windows, end_windows

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
                        ): 
    """returns a list of populated DarpEnv instances"""
    envs = []  
    from env import DarpEnv  
    for _ in range(N):
        env = DarpEnv(size, nb_requests, nb_vehicles, time_end, max_step, max_route_duration, capacity, max_ride_time, window=window)
        envs.append(env)
    return envs

def dump_training_data(envs, file: str):
    with open(file, 'wb') as handle:
        pickle.dump(envs , handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_training_data(file: str):
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

    path = "data/test_sets/generated-a2-16.pkl"
    dump_training_data(envs, path)
