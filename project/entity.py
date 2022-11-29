from abc import ABC, abstractmethod
import numpy as np
from log import logger
from typing import Optional, Tuple, List, Dict
from utils import float_equality, distance, coord2int
import pickle
from collections import deque

    
class Request():
    """Class for representing requests"""
    def __init__(self, id: int,
                pickup_position: np.ndarray,
                dropoff_position: Optional[np.ndarray],
                service_time: int,
                start_window: np.ndarray,
                end_window: Optional[np.ndarray],
                max_ride_time: int):
        self.id = id 
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        self.service_time = service_time
        self.start_window = start_window
        self.end_window = end_window
        self.original_start_window = start_window # original windows will be tightened, for evaluation we still want to use these
        self.original_end_window = end_window
        self.max_ride_time = max_ride_time
        self.state = "pickup"
        self.statedict = {"pickup": 0, "in_trunk": 1, "delivered": 2}
        self.pickup_time: float = 1440
        self.dropoff_time: float = 1440
        self.being_served: int = 0

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

    def get_min_delivery_time(self):
        return distance(self.pickup_position, self.dropoff_position)

    def tight_window(self, time_end: int, nb_requests: int):
        """make the time windows as tight as possible"""
        #filter = lambda x: min(max(0, x), time_end)
        delivery_time = int(self.get_min_delivery_time())
        self.original_start_window = self.start_window.copy()
        self.original_end_window = self.end_window.copy()
        #outbound
        if self.id < nb_requests / 2:
            #earliest i can pickup (in order to be able to reach end_window[0] within max ride time)
            self.start_window[0] = max(0, self.end_window[0] - self.max_ride_time - self.service_time)
            #latest i can pickup (in order to arrive until end_window[1])
            self.start_window[1] = min(self.end_window[1] - self.service_time, time_end)
        #inbound
        else:
            #latest i can deliver
            self.end_window[0] = max(0, self.start_window[0] + self.service_time + delivery_time)
            #earliest i can deliver
            self.end_window[1] = min(self.start_window[1] + self.service_time + self.max_ride_time, time_end)
        
        logger.debug("setting new window for: start: %s -> %s, end: %s -> %s, for %s", self.original_start_window, self.start_window, self.original_end_window, self.end_window, self)

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
        vector.append(self.pickup_position[0])
        vector.append(self.pickup_position[1])
        vector.append(self.dropoff_position[0])
        vector.append(self.dropoff_position[1])
        vector.append(self.start_window[0])
        vector.append(self.start_window[1])
        vector.append(self.end_window[0])
        vector.append(self.end_window[1])
        vector.append(self.statedict[self.state])
        vector.append(self.being_served + 1)
        return vector

    def calculate_pickup_penalty(self):
        return max(0, self.pickup_time - self.original_start_window[1])     

    def calculate_dropoff_penalty(self):
        return max(0, self.dropoff_time - self.original_end_window[1])

    def calculate_ride_time_penalty(self):
        return max(0, self.dropoff_time - (self.pickup_time + self.service_time) - self.max_ride_time)


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
        self.to_be_finished = 0 #flag changes when Vehicle decides to aim for end_depot
        self.frozen_until: float = 0.0
        self.history: List[int] = []
        self.target_request: Optional[Request] = None
        self.destination = np.empty(2)
        self.routes = deque()

    def __repr__(self):
        return f"Vehicle_{self.id}_status_{self.state}"

    def __str__(self):
        return f"Vehicle_{self.id}_status_{self.state}"

    def get_distance_to_destination(self) -> float:
        return distance(self.position, self.destination)

    def move(self, new_position: np.ndarray):
        """"set new position and save travelled distances"""
        distance_travelled = distance(self.position, new_position)
        self.total_distance_travelled += distance_travelled
        self.position = new_position
        logger.debug("%s moved to position %s", self, self.position)

    def can_pickup(self, request: Request) -> bool:
        return len(self.trunk) < self.capacity and request.state == "pickup"

    def pickup_request(self, request: Request, current_time: float):
        """given a time stamp and a request the Vehicle tries to load the request into its trunk"""
        if self.can_pickup(request):
            self.trunk.append(request)
            logger.debug("%s picked up by %s", request, self)
            request.set_state("in_trunk")
            request.being_served = self.id
            request.pickup_time = current_time
        #else:
        #    logger.debug("ERROR: %s pickup DENIED for %s", request, self)

    def can_dropoff(self, request: Request) -> bool:
        return request in self.trunk

    def dropoff_request(self, request: Request, current_time: float):
        """given a time stamp and a request the Vehicle tries to unload the request from its trunk"""
        if self.can_dropoff(request):
            self.trunk.remove(request)
            logger.debug("%s dropped of by %s", request, self)
            request.set_state("delivered")
            request.dropoff_time = current_time - request.service_time
        #else:
        #    logger.debug("ERROR: %s dropoff DENIED for %s", request, self)

    def set_state(self, state: str):
        """state of the Vehicle is either waiting, busy or finished"""
        logger.debug("setting new state: %s -> %s", self, state)
        self.state = state

    def set_destination(self, request: Request) -> np.ndarray:
        """given a Request depending on if it was picked up, the Vehicle """
        self.target_request = request
        if request in self.trunk:
            return request.dropoff_position
        else:
            return request.pickup_position
            
    def get_vector(self, requests: List[Request]) -> List[float]:
        """returns the vector representation of the Vehicle"""
        return np.array([self.position[0], self.position[1]] +
                        [distance(self.position, r.pickup_position) for r in requests] +   
                        [distance(self.position, r.dropoff_position) for r in requests] + 
                        [distance(self.position, np.array([0, 0]))])
        

    def calculate_max_route_duration_penalty(self):
       return max(0, self.total_distance_travelled - self.max_route_duration) 


class Result():
    def __init__(self, id):
        self.id = id
        self.train_loss = None
        self.test_loss = None
        self.accuracy = None
        self.policy_dict = None
        self.batch_size = None
        self.d_model = None
        self.layers = None
        self.df = None
        self.routes = 0
        self.penalty = 0

