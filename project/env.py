from logging import Logger
from typing import Optional, Tuple, List, Dict
from gym.core import RewardWrapper
import numpy as np
import gym
import torch
from utils import coord2int, float_equality, distance
from log import logger, set_level
from entity import Request, Vehicle
import generator

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
        vehicles, requests, depots = self.populate_instance()
        self.vehicles = vehicles
        self.requests = requests
        self.start_depot, self.end_depot = depots
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

    def populate_instance(self) -> Tuple[List[Vehicle], List[Request], List[np.ndarray]]:
        """"
        the function returns a list of vehicle and requests instances
        depending either by random generating or loading a cordeau instance
        """
        return generator.parse_data(self.datadir) if self.datadir else generator.generate_instance(self.seed,
                                                                                        self.size,
                                                                                        self.nb_vehicles,
                                                                                        self.nb_requests,
                                                                                        self.time_end,
                                                                                        self.capacity,
                                                                                        self.max_route_duration,
                                                                                        self.max_ride_time,
                                                                                        self.window)



    def take_action(self, action: int):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)"""
        current_vehicle = self.vehicles[self.current_vehicle]
        current_vehicle.destination =  self.destination_dict[action]
        self.already_assigned.append(action)
        logger.debug("choosing action %s (destination=%s) for %s", action, current_vehicle.destination, current_vehicle)
        current_vehicle.set_state("busy")

    def update_time_step(self, epsilon=0.1):
        "For each vehicle queries the next decision time and sets the current time attribute to the minimum of these values"
        logger.debug("Updating time step...")
        events = [self.current_time, self.time_end]
        for vehicle in self.vehicles:
            if vehicle.state != "finished":
                event = vehicle.get_distance_to_destination() 
                # if vehicle choose to not move we still increase the time by epsilon
                if float_equality(event, 0):
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
                        if any([vehicle.state != "finished" for vehicle in self.vehicles]):
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
        reward = self.penalize_broken_time_windows(reward)

        done = self.is_done()
        if done:
            logger.debug("DONE")
        # check if all Requests are delivered, if not change reward
        if done and not self.is_all_delivered():
            logger.debug("ERROR: VCEHICLES RETURNED TO DEPOT BUT REQUESTS ARE NOT DELIVERED, ABORT EPISODE")
            reward = reward + 1000
            for vehicle in self.vehicles:
                vehicle.set_state("waiting")
                self.waiting_vehicles.append(vehicle.id)
            self.current_vehicle = self.waiting_vehicles.pop()
            done = False

        if self.current_step >= self.max_step:
            reward = self.time_end * 10
            done = True
        if self.current_time >= self.time_end:
            reward = self.time_end * 10
            done = True

        observation = self.representation()
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

    def penalize_broken_time_windows(self, reward: float) -> bool:
        """checks if start, end time windows are satisfied, if not penalise"""
        start = [self.current_time > r.start_window[1] for r in self.requests if r.state == "pickup"]
        end = [self.current_time > r.end_window[1] for r in self.requests if r.state in ["pickup", "in_trunk"]]
        max_ride_time = [self.current_time - r.pickup_time > r.max_ride_time for r in self.requests if r.state == "in_trunk"] 
        max_route_duration = [v.total_distance_travelled >= self.max_route_duration for v in self.vehicles]
        reward += sum(start + end + max_route_duration + max_ride_time)
        return reward

    def nearest_action_choice(self):
        """
        Given the current environment configuration, returns the closest possible destination for the current vehicle
        among potential pickups and dropoffs
        """
        vehicle = self.vehicles[self.current_vehicle]
        choice_id, choice_dist = None, np.inf #(request id, distance to target)
        for request in self.requests:
            #potential pickup
            if vehicle.can_pickup(request, self.current_time, ignore_window=True):
                if request.id not in self.already_assigned:
                    dist = distance(vehicle.position, request.pickup_position)
                    if choice_dist > dist:
                        choice_id, choice_dist = request.id, dist

            #potential Dropoff
            elif vehicle.can_dropoff(request, self.current_time, ignore_window=True):
                dist = distance(vehicle.position, request.dropoff_position)
                if choice_dist > dist:
                    choice_id, choice_dist = request.id + self.nb_requests, dist
    
        #goto end depot
        if choice_id is None:
            choice_id = self.nb_requests * 2
        return choice_id

    #self.embed_position = nn.Embedding(2 * self.size * 100, d_model)
    #self.embed_time = nn.Embedding(1440 * 10 + 1, d_model)
    #self.embed_status = nn.Embedding(3, d_model)

    #TODO: use embeddings to have a more complex representation
    def representation(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        world = np.array([self.current_time, 
                    self.current_vehicle, 
                    self.start_depot[0],
                    self.start_depot[1],
                    self.end_depot[0],
                    self.end_depot[1],
                    self.max_ride_time,
                    self.max_route_duration])
        requests = np.stack([r.get_vector() for r in self.requests])
        if self.current_vehicle != self.nb_vehicles:
            vehicle = np.stack(self.vehicles[self.current_vehicle].get_vector())
        else: 
            vehicle = np.array([0, 0, 0, 0, 0, 0, 0])
        #w_tensor = torch.from_numpy(world).type(torch.FloatTensor).unsqueeze(dim=0)
        #r_tensor = torch.from_numpy(requests).type(torch.FloatTensor)
        #v_tensor = torch.from_numpy(vehicles).type(torch.FloatTensor).unsqueeze(dim=0)
        return world, requests, vehicle


if __name__ == "__main__":
    logger = set_level(logger, "debug")
    FILE_NAME = 'data/cordeau/a2-16.txt'
    #FILE_NAME = 'data/test_sets/t1-2.txt'
    env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1400, max_step=1000, dataset=FILE_NAME)
    #env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, dataset=FILE_NAME)
    obs = env.reset()

    #simulate env with random action samples
    rewards = []
    for t in range(1000):
        action = env.nearest_action_choice()
        #action = env.action_space.sample()
        obs, reward, done = env.step(action)
        rewards.append(reward)
        all_delivered = env.is_all_delivered()
        if done:
            print(f"Episode finished after {t + 1} steps, with reward {sum(rewards)}, all requests delivered: {all_delivered}")
            break
    delivered =  sum([request.state == "delivered" for request in env.requests])
    if not done:
        print(f"Episode finished after {t + 1} steps, with reward {rewards[-1]}, all requests delivered: {delivered}")