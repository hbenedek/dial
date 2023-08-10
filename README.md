# Dial-a-Ride

test

## Abstract

The DARP problem, also known as the Dial-a-Ride Problem, is a type of transportation problem that involves finding the most efficient way to pick up and drop off a group of passengers at different locations. The objective is to minimize the total distance traveled while satisfying the constraints of the problem, such as the capacity of the vehicles and the pickup and drop-off times for each passenger. Traditionally the prolem is formulated as mixed-integer-programming task, and solved by the branch and cut algorithm. In this project we investigated a possible, online turn-based formulation and tried to achive feasable darp solutions using supervised machine learning teqchniques and reinforcement learning, using a Transformer Encoder neuran network.

## Configuration

Clone the repository and setup conda environment.
```bash
git clone git@github.com:hbenedek/dial-a-ride.git && cd dial-a-ride
conda env create --prefix ./env --file environment.yml
conda activate ./env
```
