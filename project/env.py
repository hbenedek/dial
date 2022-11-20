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
        self.current_episode = 0
        self.penalty = 0
        self.window = window

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
        logger.debug("populate env instance with %s Vehicle and %s Request objects", self.nb_vehicles, self.nb_requests)
        if entities:
            vehicles, requests, depots = entities
            self.vehicles = vehicles
            self.requests = requests
            self.start_depot, self.end_depot = depots
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
            logger.debug("tightening window constraints for all Requests")
            for request in requests:
                request.tight_window(self.time_end, self.nb_requests)
        self.waiting_vehicles = [vehicle.id for vehicle in self.vehicles]
        self.current_vehicle = self.waiting_vehicles.pop()
        logger.debug("new current vehicle selected: %s", self.current_vehicle)
        
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



    def take_action(self, action: int):
        """the action is an id of a request or end depot"""
        current_vehicle = self.vehicles[self.current_vehicle]
        #vehicle aims at end depot
        if action == self.nb_requests:
            current_vehicle.destination = self.end_depot
        #vehicle aims at Request, depending on trunk it decides to pickup or dropoff
        else:
            target_request = self.requests[action]
            current_vehicle.destination = current_vehicle.set_destination(target_request)
            self.available_actions[action] = 0
        logger.debug("choosing action %s (destination=%s) for %s", action, current_vehicle.destination, current_vehicle)
        current_vehicle.history.append(action)
        current_vehicle.set_state("busy")

    def update_time_step(self, epsilon=0.1) -> Vehicle: #TODO: check this sometinhg is not working
        "For each vehicle queries the next decision time and sets the current time attribute to the minimum of these values"
        logger.debug("Updating time step...")
        events = dict()
        for vehicle in self.vehicles:
            if vehicle.state is "busy":
                event = vehicle.get_distance_to_destination() 
                # if vehicle choose to not move we still increase the time one
                if float_equality(event, 0):
                    event = 1
                events[vehicle] = self.current_time + event
            elif vehicle.state is "frozen":
                events[vehicle] =vehicle.frozen_until
        events = {v: e for v,e in events.items() if e > self.current_time}

        next_vehicle = min(events, key=events.get)
        new_time = events[next_vehicle]
        self.last_time_gap = new_time - self.current_time
        self.previous_time = self.current_time
        self.current_time = new_time
        logger.debug("Time step updated to %s", self.current_time)

        #unfreeze vehicles
        for vehicle in self.vehicles:
            if vehicle.state is "frozen" and self.current_time >= vehicle.frozen_until:
                vehicle.set_state("waiting")

        return next_vehicle

    def update_vehicle_position(self, vehicle: Vehicle):
        """
        update just the vehicle which the last event corresponds to
        """
        logger.debug("Updating vehicle position...")
        self.update_needed = False
         
        if vehicle.state is "busy":
            vehicle.move(vehicle.destination) 

        request = self.coordinates_dict.get(tuple(vehicle.destination)) #if destination is end depot dict returns None
        if request:
            if distance(request.pickup_position, vehicle.position) < 0.01:
                if self.current_time >= request.start_window[0]:
                    vehicle.pickup_request(request, self.current_time)
                    vehicle.set_state("waiting")
                elif vehicle.state is "busy":
                    vehicle.set_state("frozen")
                    vehicle.frozen_until = request.start_window[0]
            
            elif distance(request.dropoff_position, vehicle.position) < 0.01:
                if self.current_time >= request.end_window[0] + request.service_time:
                    vehicle.dropoff_request(request, self.current_time)
                    vehicle.set_state("waiting")
                elif vehicle.state is "busy":
                    vehicle.set_state("frozen")
                    vehicle.frozen_until = request.end_window[0] + request.service_time
        
        elif distance(self.end_depot, vehicle.position) < 0.01:
            vehicle.set_state("finished")
            self.update_needed = True
    
        if all([vehicle.state is "finished" for vehicle in self.vehicles]):
            self.update_needed = False
        elif all([vehicle.state in ["busy", "frozen", "finished"] for vehicle in self.vehicles]):
            if not self.current_time == self.time_end:
                self.update_needed = True
                   

    def mask_illegal(self) -> torch.tensor:  
        current_vehicle = self.vehicles[self.current_vehicle]
        mask = np.zeros(self.nb_requests + 1)
        #if request in trunk that action should be available
        if len(current_vehicle.trunk) == 0:
            mask[-1] = 1
        for r in current_vehicle.trunk:
            mask[r.id] = 1
        if len(current_vehicle.trunk) == current_vehicle.capacity:
            return torch.tensor(mask)
        #if trunk is empty, end depot is available 
        #if one vehicle left, it shoud deliver all remaining requests
        if sum([v.state == "finished" for v in self.vehicles]) == (self.nb_vehicles - 1) and sum([r.state == "delivered" for r in self.requests]) < self.nb_requests:
            mask[-1] = 0
        return torch.tensor(mask + self.available_actions)

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
                vehicle = self.update_time_step()
                self.update_vehicle_position(vehicle)

            logger.debug("querying waiting vehicles for new destination assignment")
            for vehicle in self.vehicles:
                if vehicle.state == 'waiting':
                    self.waiting_vehicles.append(vehicle.id)
                    logger.debug("%s added to waiting_vehicles list", self.vehicles[vehicle.id])
            next_player = self.waiting_vehicles.pop() if self.waiting_vehicles else self.nb_vehicles
        
        self.current_vehicle = next_player
        logger.debug("new current vehicle selected: %s", self.current_vehicle)

        reward = self.get_reward() 

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
        start = [round(r.calculate_pickup_penalty(), 2) for r in self.requests]
        end = [round(r.calculate_dropoff_penalty(), 2) for r in self.requests]
        max_ride_time = [round(r.calculate_ride_time_penalty(), 2) for r in self.requests] 
        max_route_duration = [round(v.calculate_max_route_duration_penalty(), 2) for v in self.vehicles]
        self.penalty = {"start_window": start, 
                        "end_window": end,
                        "max_route_duration": max_route_duration,
                        "max_ride_time": max_ride_time,
                        "sum": sum(start + end + max_route_duration + max_ride_time)}


    def nearest_action_choice(self):
        """
        Given the current environment configuration, returns the closest possible destination for the current vehicle
        among potential pickups and dropoffs
        """
        vehicle = self.vehicles[self.current_vehicle]
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
        return world, requests, vehicle


if __name__ == "__main__":
    logger = set_level(logger, "debug")
    FILE_NAME = 'data/cordeau/a2-16.txt'
    #FILE_NAME = 'data/test_sets/t1-2.txt'
    env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=1000, dataset=FILE_NAME)
    #env = DarpEnv(size=10, nb_requests=2, nb_vehicles=1, time_end=1400, max_step=100, dataset=FILE_NAME)
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


    