from typing import Optional, Tuple, List
import numpy as np
import gym

class Vehicle():
    def __init__(self, id: int, position: np.ndarray, capacity: int, max_ride_time: int):
        self.id = id
        self.position = position
        self.capacity = capacity
        self.max_ride_time = max_ride_time

class Request():
    def __init__(self, id: int, pickup_position: np.ndarray, target_position: np.ndarray, start_window: int, end_window: int, max_ride_time: int):
        self.id = id
        self.pickup_position = pickup_position
        self.target_position = target_position
        self.start_window = start_window
        self.end_window = end_window
        self.max_ride_time = max_ride_time


class DarpEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self,
                size: int,
                nb_targets: int,
                nb_drivers: int,
                max_route_duration: int,
                capacity: int,
                max_ride_time: int,
                time_end: int,
                max_step: int,
                seed: Optional[int]=None,
                dataset: Optional[str]=None):
        super(DarpEnv, self).__init__()

        self.size = size
        self.max_step = max_step
        self.nb_targets = nb_targets
        self.nb_drivers = nb_drivers
        self.max_route_duration = max_route_duration
        self.capacity = capacity
        self.max_ride_time = max_ride_time
        self.time_end = time_end
        self.action_space = gym.spaces.Discrete(nb_targets + 1)
        self.observation_space = gym.spaces.Box(low=-self.size,
                                                high=self.size,
                                                shape=(self.size + 1, self.size + 1),
                                                dtype=np.int16)
        self.current_episode = 0
        self.datadir = dataset
        self.seed = seed

        self.start_depot = None
        self.end_depot = None
        self.vehicles = []
        self.requests = []

    def populate_instance(self):
        if self.datadir:
            vehicles, requests = self.parse_data()
        else:
            vehicles, requests = self.generate_instance()
        self.vehicles = vehicles
        self.requests = requests
        
    def generate_instance(self, window=None) -> Tuple[List[Vehicle], List[Request]]:
        if self.seed:
            np.random.seed(self.seed)

        target_pickup_coodrs = np.random.uniform(-self.size, self.size, (self.nb_targets, 2))
        target_dropoff_coords = np.random.uniform(-self.size, self.size, (self.nb_targets, 2))

        #generate depot coordinates
        self.start_depot = np.random.uniform(-self.size, self.size, 2)
        self.end_depot = np.random.uniform(-self.size, self.size, 2)

        #generate time window constraints
        if window:
            pass #TODO: generate somehow time window conditions
            # (start + dist < end, 50% free start, 50% free end???)
        else:
            start_window = [0] * self.nb_targets
            end_window = [self.time_end] * self.nb_targets

        #init Driver and Target instances
        vehicles = []
        for i in range(self.nb_drivers):
            driver = Vehicle(id=i,
                            position=self.start_depot,
                            capacity=self.capacity,
                            max_ride_time=self.max_route_duration)
            vehicles.append(driver)

        requests = []
        for i in range(self.nb_targets):
            request = Request(id=i,
                            pickup_position=target_pickup_coodrs[i],
                            target_position=target_dropoff_coords[i],
                            #represents the earliest and latest time, which the service may begin
                            start_window=start_window[i],
                            end_window=end_window[i],
                            max_ride_time=self.max_ride_time)
            requests.append(request)
        return vehicles, requests
        
    def parse_data(self) -> Tuple[List[Vehicle], List[Request]]:
        file_name = self.datadir
        vehicles = []
        requests = []
        with open(file_name, 'r') as file :
            number_line = sum(1 if line and line.strip() else 0 for line in file if line.rstrip()) - 3
            file.close()

        with open(file_name, 'r') as file :
            nb_drivers, wrong_number_line, c, e, f = list(map(int, file.readline().split()))

        # Depot
        identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
        #for d in range(nb_drivers):
        #    driver = (position=np.array([X, Y]), identity=d+1, max_capacity=e, speed=1, verbose=False)
        #    drivers.append(driver)

        #for l in range(number_line) :
        #    if l < number_line//2 :
        #        identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
        #        # print(identity, X, Y, we, ty, st, en)
        #        t = Target(pickup=np.array([X, Y]), dropoff=None, start=np.array([st, en]), end=None, identity=int(identity), weight=1)
        #        targets.append(t)
        #    else :
        #        identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
        #        re_t = targets[l - number_line//2]
        #        re_t.dropoff = np.array([X, Y])
        #        re_t.end_fork = np.array([st, en])

        return vehicles, requests

    def representation(self):
        pass

    def take_action(self, action: int):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)"""
        pass

    def update_time_step(self):
        pass

    def update_drivers_position(self):
        pass
    
    
    def step(self, action):
        pass

    def print_info(self):
        pass

        




if __name__ == "__main__":
    #env = DarpEnv(size =4, nb_targets=5, nb_drivers=2, time_end=1400, max_step=100, dataset=None)
    file_name = './data/cordeau/a2-16.txt'
    with open(file_name, 'r') as file :
        number_line = sum(1 if line and line.strip() else 0 for line in file if line.rstrip()) - 3
        file.close()

    with open(file_name, 'r') as file :
        nb_drivers, nb_targets, max_route_duration, capacity, max_ride_time = list(map(int, file.readline().split()))
    
        #Depot
        _, depo_x, depo_y, _, _, _, _ = list(map(float, file.readline().split()))
        
        vehicles = []
        #Init vehicles
        for i in range(2):
            vehicle = Vehicle(id=i,
                        position=np.array([depo_x, depo_y]),
                        capacity=capacity,
                        max_ride_time=max_ride_time)
            vehicles.append(vehicle)

        #Init requests
        requests = []
        for l in range(number_line):
            if l < number_line//2:
                identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
                # print(identity, X, Y, we, ty, st, en)
                request = Request(pickup=np.array([X, Y]), dropoff=None, start=np.array([st, en]), end=None, identity=int(identity), weight=1)
                requests.append(request)
            else :
                identity, X, Y, we, ty, st, en = list(map(float, file.readline().split()))
                re_t = targets[l - number_line//2]
                re_t.dropoff = np.array([X, Y])
                re_t.end_fork = np.array([st, en])
        
    
    print(vehicles)



