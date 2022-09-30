from typing import Optional, Tuple, List
from PIL.TiffTags import _populate
import numpy as np
import gym

class Vehicle():
    def __init__(self, id: int, position: np.ndarray, capacity: int, max_route_duration: int):
        self.id = id
        self.position = position
        self.capacity = capacity
        self.max_route_duration = max_route_duration
        self.state = "waiting"

    def __repr__(self):
        return f"Vehicle_{self.id}_status:{self.state}"

    def __str__(self):
        return f"Vehicle_{self.id}_status:{self.state}"

class Request():
    def __init__(self, id: int, pickup_position: np.ndarray, dropoff_position: Optional[np.ndarray], start_window: np.ndarray, end_window: Optional[np.ndarray], max_ride_time: int):
        self.id = id
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        self.start_window = start_window
        self.end_window = end_window
        self.max_ride_time = max_ride_time
        self.state = "pickup"

    def __repr__(self):
        return f"Request_{self.id}_status:{self.state}"

    def __str__(self):
        return f"Request_{self.id}_status:{self.state}"


class DarpEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self,
                size: int,
                nb_requests: int,
                nb_vehicles: int,
                time_end: int,
                max_step: int,
                max_route_duration: Optional[int]=None,
                capacity: Optional[int]=None,
                max_ride_time: Optional[int]=None,
                seed: Optional[int]=None,
                dataset: Optional[str]=None):
        super(DarpEnv, self).__init__()

        self.size = size
        self.max_step = max_step
        self.nb_requests = nb_requests
        self.nb_vehicles = nb_vehicles
        self.max_route_duration = max_route_duration
        self.capacity = capacity
        self.max_ride_time = max_ride_time
        self.time_end = time_end
        self.action_space = gym.spaces.Discrete(nb_requests + 1)
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

        self.populate_instance()

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

        target_pickup_coodrs = np.random.uniform(-self.size, self.size, (self.nb_requests, 2))
        target_dropoff_coords = np.random.uniform(-self.size, self.size, (self.nb_requests, 2))

        #generate depot coordinates
        self.start_depot = np.random.uniform(-self.size, self.size, 2)
        self.end_depot = np.random.uniform(-self.size, self.size, 2)

        #generate time window constraints
        if window:
            pass #TODO: generate somehow time window conditions
            # (start + dist < end, 50% free start, 50% free end???)
        else:
            start_window = np.array([0, self.time_end])
            end_window = np.array([0, self.time_end])

        #init Driver and Target instances
        vehicles = []
        for i in range(self.nb_vehicles):
            driver = Vehicle(id=i,
                            position=self.start_depot,
                            capacity=self.capacity,
                            max_route_duration=self.max_route_duration)
            vehicles.append(driver)

        requests = []
        for i in range(self.nb_requests):
            request = Request(id=i,
                            pickup_position=target_pickup_coodrs[i],
                            dropoff_position=target_dropoff_coords[i],
                            #represents the earliest and latest time, which the service may begin
                            start_window=start_window,
                            end_window=end_window,
                            max_ride_time=self.max_ride_time)
            requests.append(request)
        return vehicles, requests
        
    def parse_data(self) -> Tuple[List[Vehicle], List[Request]]:
        file_name = self.datadir
        with open(file_name, 'r') as file :
            number_line = sum(1 if line and line.strip() else 0 for line in file if line.rstrip()) - 3
            file.close()

        with open(file_name, 'r') as file :
            nb_vehicles, nb_requests, max_route_duration, capacity, max_ride_time = list(map(int, file.readline().split()))

            if nb_requests != self.nb_requests:
                raise ValueError(f"DarpEnv.nb_requests={self.nb_requests} does not coincide with {nb_requests}")

            #Depot
            _, depo_x, depo_y, _, _, _, _ = list(map(float, file.readline().split()))
        
            #Init vehicles
            vehicles = []
            for i in range(nb_vehicles):
                vehicle = Vehicle(id=i,
                            position=np.array([depo_x, depo_y]),
                            capacity=capacity,
                            max_route_duration=max_route_duration)
                vehicles.append(vehicle)

            #Init requests
            requests = []
            for l in range(number_line):
                #parsing line 1, ...,n
                if l < number_line // 2:
                    identity, pickup_x, pickup_y, _, _, start_tw, end_tw = list(map(float, file.readline().split()))
              
                    request = Request(id=identity + 1,
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
    FILE_NAME = './data/cordeau/a2-16.txt'
    env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=100, dataset=FILE_NAME)
    print(env.vehicles)
    print(env.requests)
    
        
    
 


