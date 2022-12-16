from itertools import permutations
from logging import Logger
from typing import Optional, Tuple, List, Dict
import numpy as np
import gym
from requests import request
import torch
from utils import coord2int, float_equality, distance
from log import logger, set_level
from entity import Request, Vehicle
import generator

#TODO: get rid of max_step in init everywhere
class DarpEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self,
                size: int,
                nb_requests: int,
                nb_vehicles: int,
                time_end: int,
                max_route_duration: Optional[int]=None,
                capacity: Optional[int]=None,
                max_ride_time: Optional[int]=None,
                seed: Optional[int]=None,
                dataset: Optional[str]=None,
                window: Optional[bool]=None):
        super(DarpEnv, self).__init__()
        #logger.debug("initializing env")
        self.size = size
        self.max_step = nb_requests * 2 + nb_vehicles
        self.nb_requests = nb_requests
        self.nb_vehicles = nb_vehicles
        self.capacity = capacity
        self.time_end = time_end
        self.seed = seed
        self.datadir = dataset
        self.current_episode = 0
        self.penalty = 0
        self.window = window
        self.original_nb_requests = nb_requests

        if max_route_duration:
            self.max_route_duration = max_route_duration
        else:
             self.max_route_duration = min(self.max_step, self.time_end)
        if max_ride_time:
            self.max_ride_time = max_ride_time
        else:
            self.max_ride_time = min(self.max_step, self.time_end)

        self.start_depot = np.empty(2)
        self.end_depot = np.empty(2)
        self.reset()



    def reset(self, relax_window: bool=False, entities=None):
        """restarts/initializes the environment"""
        #logger.debug("populate env instance with %s Vehicle and %s Request objects", self.nb_vehicles, self.nb_requests)
        #FIXME: this could be unified somehow
        # populate instance with saved entities (used with generator.py: parse_aoyu(), model.py: reinforce())
        if entities:
            vehicles, requests, depots = entities
            self.vehicles = vehicles
            self.requests = requests
            self.start_depot, self.end_depot = depots
        # either generates new entities according to Cordeau's rules or parses a text file and returns the corresponding entities
        else:
            vehicles, requests, depots = self.populate_instance()
            self.vehicles = vehicles
            self.requests = requests
            self.start_depot, self.end_depot = depots
        if relax_window:
            logger.debug("relaxing window constraints for all Requests")
            for request in requests:
                request.relax_window(self.time_end)
        else:
            #logger.debug("tightening window constraints for all Requests")
            for request in requests:
                request.tight_window(self.time_end, self.nb_requests)
        self.depots = self.start_depot, self.end_depot
        self.waiting_vehicles = [vehicle.id for vehicle in self.vehicles]
        self.current_vehicle = self.waiting_vehicles.pop()
        #logger.debug("new current vehicle selected: %s", self.current_vehicle)
        
        self.coordinates_dict = self.coodinates_to_requests()
        # initially all actions are available, last action is only avaailable if trunk is empty (see mask_illegal())
        self.available_actions = np.ones(self.nb_requests + 1)
        self.available_actions[-1] = 0
        self.update_needed = True
        self.current_step = 0 #counts how many times the self.step() was envoked
        self.current_time = 0
        self.previous_time = 0
        self.last_time_gap = 0 #difference between consequtive time steps
        self.cumulative_reward = 0
        return self.representation()
        
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


    def mask_illegal(self, vehicle_id) -> torch.tensor:  
        current_vehicle = self.vehicles[vehicle_id]
        mask = np.zeros(self.nb_requests + 1)
        #if request in trunk that action should be available
        if len(current_vehicle.trunk) == 0:
            mask[-1] = 1
        for r in current_vehicle.trunk:
            mask[r.id] = 1
        if current_vehicle.get_trunk_load() >= current_vehicle.capacity:
            return torch.tensor(mask)
        #if trunk is empty, end depot is available 
        #if one vehicle left, it shoud deliver all remaining requests
        if sum([v.to_be_finished == 1 for v in self.vehicles]) == (self.nb_vehicles - 1) and sum([r.state == "delivered" for r in self.requests]) < self.nb_requests:
            mask[-1] = 0
        return torch.tensor(mask + self.available_actions)

    def step(self, action: int, vehicle_id: int):
        """
        moves the environment to its new state
        this involves possible new time step, vehicle position update,
        resolving pickups, dropoffs, assigning new destinations to vehicles
        """
        logger.debug("***START ENV STEP %s***", self.current_step)
        self.current_step += 1

        vehicle = self.vehicles[vehicle_id]
        #if there are still available Vehicles choose new current vehicle
        if vehicle.history:
            target_request = self.requests[vehicle.history[-1]]
            if target_request in vehicle.trunk:
                target_request.pickup_time = self.current_time
            else:
                target_request.dropoff_time = self.current_time
            vehicle.service_time = target_request.service_time
        
        if action == self.nb_requests:
            logger.debug("%s heading towards end depot", vehicle)
            travel_time = distance(vehicle.position, self.end_depot)
            vehicle.total_distance_travelled += travel_time
            vehicle.position = self.end_depot
            vehicle.free_time = self.time_end
            vehicle.to_be_finished = 1
            vehicle.history.append(action)
            vehicle.schedule.append(vehicle.free_time)
            vehicle.set_state("finished")
        else:
            target_request = self.requests[action]
            self.available_actions[action] = 0
            #print("avilable", self.available_actions)
            logger.debug("choosing action %s (destination=%s) for %s", action, vehicle.destination, vehicle)

            #perform dropoff
            if target_request in vehicle.trunk:
                travel_time = distance(vehicle.position, target_request.dropoff_position)
                vehicle.total_distance_travelled += travel_time
                window_start = target_request.end_window[0]
                vehicle.position = target_request.dropoff_position
                vehicle.dropoff_request(target_request, self.current_time)

               
            #perform pickup
            else:
                travel_time = distance(vehicle.position, target_request.pickup_position)
                vehicle.total_distance_travelled += travel_time
                window_start = target_request.start_window[0]
                vehicle.position = target_request.pickup_position
                vehicle.pickup_request(target_request, self.current_time)
              


            if vehicle.free_time + vehicle.service_time + travel_time > window_start + 1e-2:
                ride_time = vehicle.service_time + travel_time
                vehicle.free_time += ride_time
            else:
                vehicle.free_time = window_start
            vehicle.history.append(target_request.id)
            vehicle.schedule.append(vehicle.free_time)
        

        reward = self.get_reward() 

        done = self.is_done()
        if done:
            logger.debug("DONE")
        # check if all Requests are delivered, if not change reward
        if done and not self.is_all_delivered():
            logger.info("ERROR: VCEHICLES RETURNED TO DEPOT BUT REQUESTS ARE NOT DELIVERED, ABORT EPISODE")
            reward = reward + 1000
            
            for vehicle in self.vehicles:
                vehicle.set_state("waiting")
                self.waiting_vehicles.append(vehicle.id)
            self.current_vehicle = self.waiting_vehicles.pop()
            done = False

        if self.current_time >= self.time_end:
            reward = self.time_end * 10
            done = True

        observation = self.representation()
        return observation, reward, done

    def get_reward(self) -> float:
        """returns the sum of total travelled distence of all vehicles"""
        return sum([vehicle.total_distance_travelled for vehicle in self.vehicles])

    def is_done(self) -> bool:
        """checks if all Vehicles are returned to the end depot"""
        is_in_end_depot = [vehicle.state == "finished" for vehicle in self.vehicles]
        return all(is_in_end_depot)

    def is_all_delivered(self) -> bool:
        """checks if all Reuests are delivered"""
        is_delivered = [request.state == "delivered" for request in self.requests]
        return all(is_delivered)

    def penalize_broken_time_windows(self) -> Dict[str, float]:
        """checks if start, end time windows are satisfied, if not penalise linearly"""
        #delete dummy entities
        self.vehicles = [v for v in self.vehicles if v.id != self.nb_vehicles]
        self.requests = [r for r in self.requests if r.id != self.nb_requests]

        start = [round(r.calculate_pickup_penalty(), 2) for r in self.requests]
        end = [round(r.calculate_dropoff_penalty(), 2) for r in self.requests]
        max_ride_time = [round(r.calculate_ride_time_penalty(), 2) for r in self.requests] 
        max_route_duration = [round(v.calculate_max_route_duration_penalty(), 2) for v in self.vehicles]

        merged = start + end 
        tw_penalty = sum(term > 0 for term in merged)
        rt_penalty = sum(term > 0 for term in max_ride_time)
        rd_penalty = sum(term > 0 for term in max_route_duration)
        self.penalty = {"start_window": start, 
                        "end_window": end,
                        "max_route_duration": max_route_duration,
                        "max_ride_time": max_ride_time,
                        "tw_penalty": tw_penalty,
                        "rt_penalty": rt_penalty,
                        "rd_penalty": rd_penalty,
                        "sum": sum(start + end + max_route_duration + max_ride_time)}


    def nearest_action_choice(self, vehicle_id):
        """
        Given the current environment configuration, returns the closest possible destination for the current vehicle
        among potential pickups and dropoffs
        """
        vehicle = self.vehicles[vehicle_id]
        choice_id, choice_dist = None, np.inf #(request id, distance to target)
        for request in self.requests:
            #potential pickup
            if vehicle.can_pickup(request):
                if self.available_actions[request.id] == 1:
                    dist = distance(vehicle.position, request.pickup_position)
                    #if start_window is not available vehicle needs to wait
                    dist = dist + max(0, request.start_window[0] - self.current_time - dist) 
                    if choice_dist > dist:
                        choice_id, choice_dist = request.id, dist

            #potential Dropoff
            elif vehicle.can_dropoff(request):
                dist = distance(vehicle.position, request.dropoff_position)
                #if end_window is not available vehicle needs to wait
                dist = dist + max(0, request.end_window[0] - self.current_time - dist) 
                if choice_dist > dist:
                    choice_id, choice_dist = request.id, dist
    
        #goto end depot
        if choice_id is None:
            choice_id = self.nb_requests
        return choice_id

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
            curr = self.vehicles[self.current_vehicle]
            vehicle = np.stack(curr.get_vector(self.requests))
        else: 
            vehicle = np.array([0] * (2 * self.nb_requests + 1))
        return world, requests, vehicle


    def pad_env(self, nb_vehicles: int, nb_requests: int):
        """ populate instance with dummy Vehicles and Requests"""
        request_diff = nb_requests - self.nb_requests
        vehicle_diff = nb_vehicles - self.nb_vehicles
        for _ in range(vehicle_diff):
            dummy_vehicle = Vehicle(nb_vehicles, np.array([0.0,0.0]), 0, 0)
            dummy_vehicle.state = "finished"
            dummy_vehicle.to_be_finished = 1
            self.vehicles.append(dummy_vehicle)
            self.nb_vehicles += 1

        for _ in range(request_diff):
            dummy_request = Request(self.nb_requests, np.array([0.0,0.0]), np.array([0.0,0.0]), 0, np.array([0.0,0.0]), np.array([0.0,0.0]), 0)
            dummy_request.pickup_time = 0
            dummy_request.dropoff_time = 0
            dummy_request.state = "delivered"
            self.requests.append(dummy_request)
            self.nb_requests += 1
        self.available_actions = np.pad(self.available_actions, (0, nb_requests + 1 - len(self.available_actions)), "constant")
        #self.max_step = 2 * self.nb_requests + self.nb_vehicles
        logger.debug("env padded to %s Vehicles and %s Requests", self.nb_vehicles, self.nb_requests)

    def augment(self, permutation: np.ndarray):
        self.requests = [self.requests[i] for i in permutation]
        self.available_actions = self.available_actions[permutation]


if __name__ == "__main__":

    ###################################    EXAMPLE USAGE 1 (ENV SIMULATOR)   ##########################################

    logger = set_level(logger, "debug")

    # loading and initializing a darp environment
    FILE_NAME = 'data/cordeau/a2-16.txt'
    env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=1000, dataset=FILE_NAME)
    obs = env.representation()
    
    #simulate env with nearest neighbor action
    rewards = []
    for t in range(100):
        action = env.nearest_action_choice()    
        obs, reward, done = env.step(action)
        rewards.append(reward)
        delivered =  sum([request.state == "delivered" for request in env.requests])
        all_delivered = env.is_all_delivered()
        if done:
            break

    env.penalize_broken_time_windows()

    # print out results
    total = sum([v.total_distance_travelled for v in env.vehicles])
    logger.info(f"Episode finished after {t + 1} steps, with reward {total}")
    for vehicle in env.vehicles:
        logger.info(f'{vehicle} history: {vehicle.history}')
    delivered =  sum([request.state == "delivered" for request in env.requests])
    in_trunk = sum([r.state == "in_trunk" for r in env.requests])
    pickup = sum([r.state == "pickup" for r in env.requests])
    logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
    logger.info(f'delivered: {delivered}, in trunk: {in_trunk}, waiting: {pickup}')
    logger.info("*** PENALTY ***")
    logger.info("start_window: %s", env.penalty["start_window"])
    logger.info("end_window: %s", env.penalty["end_window"])
    logger.info("max_route_duration: %s", env.penalty["max_route_duration"])
    logger.info("max_ride_time: %s", env.penalty["max_ride_time"])
    logger.info("total penalty: %s", env.penalty["sum"])



    