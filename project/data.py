from env import DarpEnv
from typing import List
import pickle


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
