from logging import Logger
from typing import Optional, Tuple, List, Dict
import numpy as np
import gym
from numpy.random.mtrand import set_state
import torch
from utils import coord2int, float_equality, distance
from log import logger, set_level

class Request():
    """Class for representing requests"""
    def __init__(self, id: int,
                pickup_position: np.ndarray,
                dropoff_position: Optional[np.ndarray],
                start_window: np.ndarray,
                end_window: Optional[np.ndarray],
                max_ride_time: int):
        self.id = id - 1
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        self.start_window = start_window
        self.end_window = end_window
        self.max_ride_time = max_ride_time
        self.state = "pickup"
        self.statedict = {"pickup": 0, "in_trunk": 1, "delivered": 2}
        self.pickup_time: Optional[float] = None

    def __repr__(self):
        return f"Request_{self.id}_status_{self.state}"

    def __str__(self):
        return f"Request_{self.id}_status_{self.state}"

    def check_window(self, current_time: float, start: bool) -> bool:
        """
        given the current time, the function cheks whether the time window conditions
        (start or end indicated by boolean flag) are met for the Request
        """
        if start:
            start, end = self.start_window
        else:
            start, end = self.end_window
        if float_equality(current_time, start) or float_equality(current_time, end):
            return True
        return start <= current_time and end >= current_time

    def set_state(self, state: str):
        """state of the Request is either pickup, in_trunk or delivered"""
        logger.debug( "setting new state: %s -> %s", self, state)
        self.state = state

    def get_service_time(self):
        return distance(self.pickup_position, self.dropoff_position)

    def tight_window(self):
        """make the time windows as tight as possible"""
        service_time = self.get_service_time()
        #earliest i can deliver
        self.end_window[0] = max(self.end_window[0], self.start_window[0] + service_time)
        #latest i can deliver
        self.end_window[1] = min(self.end_window[1], self.start_window[1] + self.max_ride_time)
        #earliest i can pickup (in order to be able to reach end_window[0] within max ride time)
        self.start_window[0] = max(self.start_window[0], self.end_window[0] - self.max_ride_time)
        #latest i can pickup (in order to arrive until end_window[1])
        self.start_window[1] = min(self.start_window[1], self.end_window[1] - service_time)

    def relax_window(self, time_end: int):
        """drop all window constrains"""
        #earliest i can deliver
        self.end_window[0] = 0
        #latest i can deliver
        self.end_window[1] = time_end
        #earliest i can pickup (in order to be able to reach end_window[0] within max ride time)
        self.start_window[0] = 0
        #latest i can pickup (in order to arrive until end_window[1])
        self.start_window[1] = time_end
        logger.debug("setting new window for: start: %s, end: %s, for %s", self.start_window, self.end_window, self)


    def get_vector(self) -> List[int]:
        """returns the vector representation of the Request"""
        vector = [self.id]
        vector.append(coord2int(self.pickup_position[0]))
        vector.append(coord2int(self.pickup_position[1]))
        vector.append(coord2int(self.dropoff_position[0]))
        vector.append(coord2int(self.dropoff_position[1]))
        vector.append(self.start_window[0])
        vector.append(self.start_window[1])
        vector.append(self.end_window[0])
        vector.append(self.end_window[1])
        vector.append(self.statedict[self.state])
        vector.append(self.max_ride_time)
        return vector


class Vehicle():
    """Class for representing Vehicles"""
    def __init__(self, id: int, position: np.ndarray, capacity: int, max_route_duration: int):
        self.id = id
        self.position = position
        self.capacity = capacity
        self.max_route_duration = max_route_duration
        self.state: str = "waiting"
        self.statedict = {"waiting": 0, "busy": 1, "finished": 2}
        self.trunk: List[Request] = []
        self.dist_to_destination: float = 0
        self.last_distance_travelled = 0
        self.total_distance_travelled = 0
        self.destination = np.empty(2)

    def __repr__(self):
        return f"Vehicle_{self.id}_status_{self.state}"

    def __str__(self):
        return f"Vehicle_{self.id}_status_{self.state}"

    def get_distance_to_destination(self) -> float:
        return distance(self.position, self.destination)

    def move(self, new_position: np.ndarray):
        """"set new position and save travelled distances"""
        distance_travelled = distance(self.position, new_position)
        self.last_distance_travelled = distance_travelled
        self.total_distance_travelled += distance_travelled
        self.position = new_position
        logger.debug("%s moved to position %s", self, self.position)

    def can_pickup(self, request: Request, current_time: float) -> bool:
        dist = distance(self.position, request.pickup_position)
        return len(self.trunk) < self.capacity and request.check_window(current_time + dist, start=True) and request.state == "pickup"

    def pickup_request(self, request: Request, current_time: float):
        """given a time stamp and a request the Vehicle tries to load the request into its trunk"""
        if self.can_pickup(request, current_time):
            self.trunk.append(request)
            logger.debug("%s picked up by %s", request, self)
            request.set_state("in_trunk")
            request.pickup_time = current_time
        else:
            logger.debug("ERROR: %s pickup DENIED for %s", request, self)

    def can_dropoff(self, request: Request, current_time: float) -> bool:
        dist = distance(self.position, request.dropoff_position)
        return request.check_window(current_time + dist, start=False) and request in self.trunk

    def dropoff_request(self, request: Request, current_time: float):
        """given a time stamp and a request the Vehicle tries to unload the request from its trunk"""
        if self.can_dropoff(request, current_time):
            self.trunk.remove(request)
            logger.debug("%s dropped of by %s", request, self)
            request.set_state("delivered")
        else:
            logger.debug("ERROR: %s dropoff DENIED for %s", request, self)

    def set_state(self, state: str):
        """state of the Vehicle is either waiting, busy or finished"""
        logger.debug( "setting new state: %s -> %s", self, state)
        self.state = state

    def get_vector(self) -> List[int]:
        """returns the vector representation of the Request"""
        vector = [self.max_route_duration,
                coord2int(self.position[0]),
                coord2int(self.position[1]),
                self.statedict[self.state]]
        trunk =  [r.id for r in self.trunk]
        trunk = trunk + [0] * (self.capacity - len(self.trunk))
        return vector + trunk

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
                dataset: Optional[str]=None,
                window: Optional[bool]=None):
        super(DarpEnv, self).__init__()
        logger.debug("initializing env")
        self.size = size
        self.max_step = max_step
        self.nb_requests = nb_requests
        self.nb_vehicles = nb_vehicles
        self.capacity = capacity
        self.time_end = time_end
        self.seed = seed
        self.datadir = dataset
        # we probably don't need gym, but let's keep it here.. maybe for visualization purposes...
        self.action_space = gym.spaces.Discrete(nb_requests * 2 + 1)
        self.observation_space = gym.spaces.Box(low=-self.size,
                                                high=self.size,
                                                shape=(self.size + 1, self.size + 1),
                                                dtype=np.int16)
        self.current_episode = 0
        self.window = window

        if max_route_duration:
            self.max_route_duration = max_route_duration
        else:
             self.max_route_duration = self.max_step
        if max_ride_time:
            self.max_ride_time = max_ride_time
        else:
            self.max_ride_time = self.max_step

        self.start_depot = np.empty(2)
        self.end_depot = np.empty(2)
        self.reset()

    def reset(self, relax_window: bool=False):
        """restarts/initializes the environment"""
        logger.debug("populate env instance with %s Vehicle and %s Request objects", self.nb_vehicles, self.nb_requests)
        vehicles, requests = self.populate_instance()
        self.vehicles = vehicles
        self.requests = requests
        if relax_window:
            logger.debug("relaxing window constraints for all Requests")
            for request in requests:
                request.relax_window(self.time_end)
        else:
            logger.debug("tightening window constraints for all Requests")
            for request in requests:
                request.tight_window()
        self.waiting_vehicles = [vehicle.id for vehicle in self.vehicles]
        self.current_vehicle = self.waiting_vehicles.pop()
        logger.debug("new current vehicle selected: %s", self.current_vehicle)
        self.destination_dict = self.output_to_destination()
        self.coordinates_dict = self.coodinates_to_requests()
        self.already_assigned: List[int] = []
        self.update_needed = True
        self.current_step = 0 #counts how many times the self.step() was envoked
        self.current_time = 0
        self.last_time_gap = 0 #difference between consequtive time steps
        self.cumulative_reward = 0
        return self.representation()
        
    def output_to_destination(self) -> Dict[int, np.ndarray]:
        """"
        the Transformer, given a state configuration, for the current player outputs a probability distribution of the
        next potential target nodesthis function converts the output index to destination coordinates
        """
        pickups = {i: r.pickup_position for i, r in enumerate(self.requests)}
        dropoffs = {i + self.nb_requests: r.dropoff_position for i, r in enumerate(self.requests)}
        depots = {self.nb_requests * 2: self.end_depot}
        return {**pickups, **dropoffs, **depots}

    def coodinates_to_requests(self) -> Dict[Tuple[float, float], Request]:
        """converts the 2D coordinates to its corresponding request"""
        pickups = {tuple(r.pickup_position): r for r in self.requests}
        dropoffs = {tuple(r.dropoff_position): r for r in self.requests}
        return {**pickups, **dropoffs}

    def populate_instance(self) -> Tuple[List[Vehicle], List[Request]]:
        """"
        the function returns a list of vehicle and requests instances
        depending either by random generating or loading a cordeau instance
        """
        return self.parse_data() if self.datadir else self.generate_instance(self.window)

    def generate_instance(self, window=None) -> Tuple[List[Vehicle], List[Request]]:
        """"generates random pickup, dropoff and other constraints, a list parsed of Vehicle and Request objects"""
        if self.seed:
            np.random.seed(self.seed)

        target_pickup_coodrs = np.random.uniform(-self.size, self.size, (self.nb_requests, 2))
        target_dropoff_coords = np.random.uniform(-self.size, self.size, (self.nb_requests, 2))

        #generate depot coordinates
        self.start_depot = np.random.uniform(-self.size, self.size, 2)
        self.end_depot = np.random.uniform(-self.size, self.size, 2)

        #generate time window constraints
        if window:
            start_windows, end_windows  = self.generate_window()
        else:
            start_windows = [np.array([0, self.time_end]) for _ in range(self.nb_requests)]
            end_windows = [np.array([0, self.time_end]) for _ in range(self.nb_requests)]

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
                            start_window=start_windows[i],
                            end_window=end_windows[i],
                            max_ride_time=self.max_ride_time)
            requests.append(request)
        return vehicles, requests

    def parse_data(self) -> Tuple[List[Vehicle], List[Request]]:
        """given a cordeau2006 instance, the function returns a list parsed of Vehicle and Request objects"""
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
            self.start_depot = np.array([depo_x, depo_y])
            self.end_depot = np.array([depo_x, depo_y])

            #Init vehicles
            vehicles = []
            for i in range(nb_vehicles):
                vehicle = Vehicle(id=i,
                            position=self.start_depot,
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

        return vehicles, requests

    def take_action(self, action: int):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)"""
        current_vehicle = self.vehicles[self.current_vehicle]
        current_vehicle.destination =  self.destination_dict[action]
        self.already_assigned.append(action)
        logger.debug("choosing action %s (destination=%s) for %s", action, current_vehicle.destination, current_vehicle)
        current_vehicle.set_state("busy")

    def update_time_step(self, epsilon=0.01):
        "For each vehicle queries the next decision time and sets the current time attribute to the minimum of these values"
        logger.debug("Updating time step...")
        events = [self.current_time, self.time_end]
        for vehicle in self.vehicles:
            if vehicle.state != "finished":
                event = vehicle.get_distance_to_destination() 
                # if vehicle choose to not move we still increase the time by epsilon
                if event == 0:
                    event = epsilon
                events.append(self.current_time + event)

        events = [event for event in events if event > self.current_time]
        new_time = min(events)
        self.last_time_gap = new_time - self.current_time
        self.current_time = new_time
        logger.debug("Time step updated to %s", self.current_time)

    def update_vehicle_position(self):
        """
        all vehicles are moved closer to their current destination
        pickups and dropoffs are resolved
        """
        logger.debug("Updating vehicle positions...")
        self.update_needed = False
        for vehicle in self.vehicles:
            if vehicle.state != "finished":
                dist_to_destination = vehicle.get_distance_to_destination()
                if float_equality(self.last_time_gap, dist_to_destination, eps=0.001):
                    #vehicle arriving to destination
                    vehicle.move(vehicle.destination)

                    #resolving pickup, dropoff or depot arrival
                    request = self.coordinates_dict.get(tuple(vehicle.destination)) #if destination is end depot dict returns None
                    if request:
                        if np.array_equal(request.pickup_position, vehicle.position):
                            vehicle.pickup_request(request, self.current_time)
                            vehicle.set_state("waiting")
                        elif np.array_equal(request.dropoff_position, vehicle.position):
                            vehicle.dropoff_request(request, self.current_time)
                            vehicle.set_state("waiting")
                    elif np.array_equal(self.end_depot, vehicle.position):
                        vehicle.set_state("finished")
                        self.update_needed = True

                #move vehicle closer to its destination
                elif self.last_time_gap < dist_to_destination:
                    unit_vector = (vehicle.destination - vehicle.position) / dist_to_destination
                    new_position = vehicle.position + self.last_time_gap * unit_vector
                    vehicle.move(new_position)

                else:
                    vehicle.set_state("waiting")


    def mask_illegal(self) -> torch.tensor:  
        current_vehicle = self.vehicles[self.current_vehicle]
        trunk = [r.id for r in current_vehicle.trunk]
        pickups = [1 if r.state == "pickup" else 0 for r in self.requests]
        dropoffs = [1 if r.id in trunk else 0 for r in self.requests]
        if sum(dropoffs) > 0:
            end = [0]
        else:
            end = [1]
        return torch.tensor(pickups + dropoffs + end)

    def step(self, action: int):
        """
        moves the environment to its new state
        this involves possible new time step, vehicle position update,
        resolving pickups, dropoffs, assigning new destinations to vehicles
        """
        logger.debug("***START ENV STEP %s***", self.current_step)
        self.take_action(action)
        self.current_step += 1

        #if there are still available Vehicles choose new current vehicle
        next_player = self.waiting_vehicles.pop() if self.waiting_vehicles else self.nb_vehicles
        if next_player == self.nb_vehicles:
            self.update_needed = True #if a Vehicle arrives at end depot we need to perform one more update
            while self.update_needed:
                self.update_time_step()
                self.update_vehicle_position()

            logger.debug("querying waiting vehicles for new destination assignment")
            for vehicle in self.vehicles:
                if vehicle.state == 'waiting':
                    self.waiting_vehicles.append(vehicle.id)
                    logger.debug("%s added to waiting_vehicles list", self.vehicles[vehicle.id])
            next_player = self.waiting_vehicles.pop() if self.waiting_vehicles else self.nb_vehicles
        
        self.current_vehicle = next_player
        logger.debug("new current vehicle selected: %s", self.current_vehicle)

        reward = self.get_reward() 

        #TODO: this part if a bit confusing (vehicle stays at depot, does not move and happily accepts 0 final reward)
        observation = self.representation()
        if self.are_time_windows_satisfied():
        #if True:
            #check if vehicles returned to end depot
            done = self.is_done()
        else:
            logger.debug("ERROR: TIME WINDOW CONSTRAINTS ARE VIOLATED, ABORT EPISODE")
            done = True
            reward = reward + self.time_end * 4
        # check if all Requests are delivered, if not change reward
        if done and not self.is_all_delivered():
            logger.debug("ERROR: VCEHICLES RETURNED TO DEPOT BUT REQUESTS ARE NOT DELIVERED, ABORT EPISODE")
            reward = reward + self.time_end * 4
        elif self.current_step == self.max_step:
            reward = self.time_end * 4
        return observation, reward, done

    def get_reward(self) -> float:
        """returns the sum of total travelled distence of all vehicles"""
        #TODO: maybe we need incremental rewards as well? depends on how we use the RL learning signal
        return sum([vehicle.last_distance_travelled for vehicle in self.vehicles])

    def is_done(self) -> bool:
        """checks if all Vehicles are returned to the end depot"""
        is_in_end_depot = [vehicle.state == "finished" for vehicle in self.vehicles]
        return all(is_in_end_depot)

    def is_all_delivered(self) -> bool:
        """checks if all Reuests are delivered"""
        is_delivered = [request.state == "delivered" for request in self.requests]
        return all(is_delivered)

    def are_time_windows_satisfied(self) -> bool:
        """checks if start, end time windows are satisfied"""
        start = [self.current_time <= r.start_window[1] for r in self.requests if r.state == "pickup"]
        end = [self.current_time <= r.end_window[1] for r in self.requests if r.state in ["pickup", "in_trunk"]]
        max_ride_time = [self.current_time - r.pickup_time < r.max_ride_time for r in self.requests if r.state == "in_trunk"] 
        max_route_duration = [v.total_distance_travelled <= self.max_route_duration for v in self.vehicles]
        return all(start + end + max_route_duration + max_ride_time)

    def next_observation(self):
        """returns the input (word, vehicle and request info) for the Transformer model for the next env step"""
        pass

    def generate_window(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        #FIXME: this method does not seem very sophisticated, copied from Pawal's code
        """Generate 50% of free dropof conditions, and 50% of free pickup time conditions time windows for Requests"""
        start_windows = []
        end_windows = []
        for j in range(self.nb_requests):
            # Generate start and end point for window
            start = np.random.randint(0, self.time_end * 0.9) 
            end = np.random.randint(start + 15, start + 45)
            #free pickup condition
            if j < self.nb_requests // 2:
                start_fork = [max(0, start - self.max_ride_time), end]
                end_fork = [start, end]
            #free dropoff condition
            else:
                start_fork = [start, end]
                end_fork = [start, min(self.time_end, end + self.max_ride_time)]

            start_windows.append(np.array(start_fork))
            end_windows.append(np.array(end_fork))
        return start_windows, end_windows

    def nearest_action_choice(self):
        """
        Given the current environment configuration, returns the closest possible destination for the current vehicle
        among potential pickups and dropoffs
        """
        vehicle = env.vehicles[self.current_vehicle]
        choice_id, choice_dist = None, np.inf #(request id, distance to target)
        for request in self.requests:
            #potential pickup
            if vehicle.can_pickup(request, self.current_time):
                if request.id not in self.already_assigned:
                    dist = distance(vehicle.position, request.pickup_position)
                    if choice_dist > dist:
                        choice_id, choice_dist = request.id, dist

            #potential Dropoff
            elif vehicle.can_dropoff(request, self.current_time):
                dist = distance(vehicle.position, request.dropoff_position)
                if choice_dist > dist:
                    choice_id, choice_dist = request.id + self.nb_requests, dist
    
        #goto end depot
        if choice_id is None:
            choice_id = env.nb_requests * 2
        return choice_id


    def representation(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        world = np.array([self.current_time, self.current_vehicle, coord2int(self.start_depot[1]), coord2int(self.end_depot[1])])
        requests = np.stack([r.get_vector() for r in self.requests])
        if self.current_vehicle != self.nb_vehicles:
            vehicles = np.stack(self.vehicles[self.current_vehicle].get_vector())
        else: 
            vehicles = np.array([0, 0, 0, 0, 0, 0, 0])
        w_tensor = torch.from_numpy(world).type(torch.FloatTensor).unsqueeze(dim=0)
        r_tensor = torch.from_numpy(requests).type(torch.FloatTensor)
        v_tensor = torch.from_numpy(vehicles).type(torch.FloatTensor).unsqueeze(dim=0)
        return w_tensor, r_tensor, v_tensor


if __name__ == "__main__":
    #FILE_NAME = './data/cordeau/a2-16.txt'
    logger = set_level(logger, "debug")
    FILE_NAME = '../data/test_sets/t1-2.txt'
    #env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=100, dataset=FILE_NAME)
    env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, dataset=FILE_NAME)
    obs = env.reset()

    #simulate env with random action samples
    cum_reward = 0
    for t in range(100):
        action = env.nearest_action_choice()
        #action = env.action_space.sample()
        obs, reward, done = env.step(action)
        cum_reward += reward
        all_delivered = env.is_all_delivered()
        if done:
            print(f"Episode finished after {t + 1} steps, with reward {cum_reward}, all requests delivered: {all_delivered}")
            break