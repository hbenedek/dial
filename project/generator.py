from typing import List, Tuple
import pickle
import urllib
import requests
import numpy as np
from bs4 import BeautifulSoup
import re
from entity import Request, Vehicle
from collections import deque
from log import logger, set_level
import json
import glob


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


def generate_instance(seed: int,
                    size: int, 
                    nb_vehicles: int, 
                    nb_requests: int, 
                    time_end: int, 
                    capacity: int, 
                    max_route_duration: int, 
                    max_ride_time: int,
                    window=True,
                    random_depot=False) -> Tuple[List[Vehicle], List[Request], List[np.ndarray]]:
        """"generates random pickup, dropoff and other constraints, a list parsed of Vehicle and Request objects"""
        if seed:
            np.random.seed(seed)

        target_pickup_coodrs = np.random.uniform(-size, size, (nb_requests, 2))
        target_dropoff_coords = np.random.uniform(-size, size, (nb_requests, 2))

        #generate depot coordinates 
        if random_depot:
            start_depot = np.random.uniform(-size, size, 2)
            end_depot = np.random.uniform(-size, size, 2)
        else:
            start_depot = np.array([0,0])
            end_depot = np.array([0,0])

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
        for i in range(0, nb_requests):
            request = Request(id=i + 1,
                            pickup_position=target_pickup_coodrs[i],
                            dropoff_position=target_dropoff_coords[i],
                            service_time = 3,
                            #represents the earliest and latest time, which the service may begin
                            start_window=start_windows[i],
                            end_window=end_windows[i],
                            max_ride_time=max_ride_time)
            requests.append(request)
            #logger.debug("setting new window - start: %s, end: %s, for %s", request.start_window, request.end_window, request)
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
                    identity, pickup_x, pickup_y, service_time, _, start_tw, end_tw = list(map(float, file.readline().split()))

                    request = Request(id=int(identity),
                                pickup_position=np.array([pickup_x, pickup_y]),
                                dropoff_position=None,
                                service_time = service_time,
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
        """
        Generate 50% of free dropoff conditions, and 50% of free pickup time conditions time windows for Requests
        according to Cordeau's rules
        """
        start_windows = []
        end_windows = []
        for j in range(nb_requests):
            #free pickup condition (outbound)
            if j < nb_requests // 2:
                end = np.random.randint(0, time_end - 60)
                start = end - 15 

                start_fork = [0, time_end]
                end_fork = [start, end]
            #free dropoff condition (inbound)
            else:
                start = np.random.randint(60, time_end)
                end = start + 15 

                start_fork = [start, end]
                end_fork = [0, time_end]

            start_windows.append(np.array(start_fork))
            end_windows.append(np.array(end_fork))
        return start_windows, end_windows


def dump_data(object, file: str):
    with open(file, 'wb') as handle:
        pickle.dump(object , handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(file: str):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def generate_environments(
                        N: int, #number of instances to generate
                        size: int, 
                        nb_vehicles: int,
                        nb_requests: int,
                        time_end: int,
                        max_step: int,
                        max_route_duration: int,
                        capacity: int,
                        max_ride_time: int,
                        window: bool): 
    """returns a list of populated DarpEnv instances"""
    envs = []  
    from env import DarpEnv  
    for i in range(N):
        if i % 100 == 0:
            logger.info("%s environments generated", i)
        env = DarpEnv(size, nb_requests, nb_vehicles, time_end, max_step, max_route_duration, capacity, max_ride_time, window=window)
        envs.append(env)
    return envs

###################################    PARSER AND LOADER FOR AOYU'S DATASET    ##########################################

def parse_aoyu(datadir: str):
    envs = []
    with open(datadir, 'r') as file:
        for pair in file:
            pair = json.loads(pair)
            nb_vehicles = pair['instance'][0][0]
            nb_requests = pair['instance'][0][1]
            max_route_duration = pair['instance'][0][2]
            capacity = pair['instance'][0][3]
            max_ride_time = pair['instance'][0][4]
            objective = pair['objective']

            requests = []
            for i in range(0, 2* (nb_requests + 1)):
                node = pair['instance'][i + 1]
                if i == 0:
                    start_depot = np.array([float(node[1]), float(node[2])])
                    continue
                elif i == 2 * (nb_requests + 1) - 1:
                    end_depot = np.array([float(node[1]), float(node[2])])
                    continue
                elif i <= nb_requests:
                    request = Request(id=int(i),
                                    pickup_position=np.array([float(node[1]), float(node[2])]),
                                    dropoff_position=None,
                                    service_time =  node[3],
                                    #represents the earliest and latest time, which the service may begin
                                    start_window=np.array([float(node[5]), float(node[6])]),
                                    end_window=None,
                                    max_ride_time=max_ride_time)
                    requests.append(request)
                else:
                    # Drop-off nodes
                    request = requests[i - nb_requests - 1]
                    request.dropoff_position = np.array([float(node[1]), float(node[2])])
                    request.end_window = np.array([float(node[5]), float(node[6])])    

                
            #init Driver and Target instances
            vehicles = []
            length = len(pair['routes'])
            for i in range(0,nb_vehicles):
                vehicle = Vehicle(id=i,
                            position=start_depot,
                            capacity=capacity,
                            max_route_duration=max_route_duration)
                if length > i:
                    vehicle.routes = deque(pair['routes'][i] + [nb_requests * 2 + 1])
                    vehicle.schedule = pair['schedule'][i]
                else:
                    vehicle.routes = deque([nb_requests * 2 + 1])
                    vehicle.schedule = []
                vehicles.append(vehicle)

            depots = [start_depot, end_depot]
            from env import DarpEnv 
            env = DarpEnv(10, nb_requests, nb_vehicles, 1440, 2 * nb_requests + nb_vehicles, max_route_duration, capacity, max_ride_time)
            env.objective = objective
            env.reset(entities=[vehicles, requests, depots])        
            envs.append(env)
        return envs


def load_aoyo(instance: str):
    train_path, test_path = f"data/aoyu/{instance}-train.txt", f"data/aoyu/{instance}-test.txt"
    logger.info("parsing %s for training data", train_path)
    train_envs = parse_aoyu(train_path)
    logger.info("parsing %s for testing data", test_path)
    test_envs = parse_aoyu(test_path)
    return train_envs, test_envs


if __name__ == "__main__":
    logger = set_level(logger, "info")
    #envs = generate_training_data(N=10000,
    #                    size= 10, 
    #                    nb_vehicles=2,
    #                    nb_requests=16,
    #                    time_end=1440,
    #                    max_step=1000,
    #                    max_route_duration=480,
    #                    capacity=3,
    #                    max_ride_time=30,
    #                    window=True)

    
    #logger.info("data dump starts...")
    #path = "data/processed/generated-10000-a2-16.pkl"
    #dump_data(envs, path)
    #logger.info("data successfully dumped")

    envs = parse_aoyu("data/aoyu/a2-16-test.txt")
    print("DONE")
    logger = set_level(logger, "debug")
    env = envs[0]
    obs = env.representation()
    #simulate env with nearest neighbor action
    rewards = []
    for t in range(34):
        current = env.vehicles[env.current_vehicle]
        action = current.routes.popleft() - 1
        if action >= env.nb_requests:
            action = action - env.nb_requests
        obs, reward, done = env.step(action)
        rewards.append(reward)
        delivered =  sum([request.state == "delivered" for request in env.requests])
        all_delivered = env.is_all_delivered()
        if done:
            break
    env.penalize_broken_time_windows()

    total = sum([v.total_distance_travelled for v in env.vehicles])
    logger.info(f"Episode finished after {t + 1} steps, with reward {total}")
    for vehicle in env.vehicles:
        logger.info(f'{vehicle} history: {vehicle.history}')
    delivered =  sum([request.state == "delivered" for request in env.requests])
    in_trunk = sum([r.state == "in_trunk" for r in env.requests])
    pickup = sum([r.state == "pickup" for r in env.requests])
    logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
    logger.info("*** PENALTY ***")
    logger.info("start_window: %s", env.penalty["start_window"])
    logger.info("end_window: %s", env.penalty["end_window"])
    logger.info("max_route_duration: %s", env.penalty["max_route_duration"])
    logger.info("max_ride_time: %s", env.penalty["max_ride_time"])
    logger.info("total penalty: %s", env.penalty["sum"])