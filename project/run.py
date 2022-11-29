import torch
from model import Policy
from utils import get_device
from log import logger
from supervised import supervised_trainer
from evaluate import evaluate_model
from env import DarpEnv

instance="a2-16"
result_path = "models"
supervised_policy="rf"
trial = "06"
batch_size = 256
nb_epochs = 10
id = f"result-{instance}-supervised-{supervised_policy}-{trial}-aoyu128"
FILE_NAME = f'data/cordeau/{instance}.txt'
test_env = DarpEnv(size=10, nb_requests=16, nb_vehicles=2, time_end=1440, max_step=34, dataset=FILE_NAME)

#initialize policy
policy = Policy(d_model=256, nhead=8, nb_requests=16, nb_vehicles=2, num_layers=4, time_end=1440, env_size=10)
PATH = "models/result-a2-16-supervised-rf-05-aoy4layer"
r = load_data(PATH)
state = r.policy_dict
policy.load_state_dict(state)

device = get_device()
policy = policy.to(device)
logger.info("training on device: %s", device)

#initialize optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

#start train
result = supervised_trainer(id, 
                        instance,
                        result_path,
                        supervised_policy,
                        batch_size, 
                        nb_epochs, 
                        policy,
                        optimizer) 


# passing the model to CUDA if available 
policy.eval()
routing_cost, window_penalty, delivered = evaluate_model(policy, test_env)



