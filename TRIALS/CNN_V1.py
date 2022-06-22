import sys

# PyTorch
import torch
import torch.optim as optim

sys.path.append("../ActorCritic") # to import ActorCritic method, models
sys.path.append("../") # to import Maze_Solver

# import model
from models.CNN import CNN_V1
from ActorCritic import ActorCritic

# Environment 
from Maze_Solver import MazeSolverEnv

MAX_EPISODES = 3000
MAX_TIMESTEPS = 1000

ALPHA = 3e-4 # learning rate
GAMMA = 0.99 # step-size

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = MazeSolverEnv()

# set ActorCritic
num_actions = env.num_action
num_states = env.num_obs
ACmodel = CNN_V1(num_states[0], num_states[1], num_actions).to(device)
optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

ActorCritic_parameters = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': ACmodel, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'stepsize': GAMMA # step-size for updating Q value
}

# Initialize Actor-Critic Mehtod
AC = ActorCritic(**ActorCritic_parameters)

# TRAIN Agent
AC.train(MAX_EPISODES)

