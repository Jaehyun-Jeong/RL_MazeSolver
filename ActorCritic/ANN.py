from __future__ import print_function

import numpy as np
import random
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import pygame
import Maze_Solver as maze_solver
from Maze_Solver import MazeSolver, MazeSolverEnv
import Maze_Generator as maze_generator

class ActorCritic(nn.Module):
    def __init__(self, **params) , inputs, outputs, env, model, optimizer, MAX_EPISODES, MAX_TIMESTEPS, learning_rate, step_size):
        super(ActorCritic, self).__init__()
        
        # for Actor
        self.actor_fc1 = nn.Linear(params[''], 256)
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.critic_fc2 = nn.Linear(256, 1)
        self.head = nn.Softmax(dim=0)
        
        # torch.log makes nan(not a number) error so we have to add some small number in log function
        self.ups=1e-7

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        state = x.to(device)
        
        probs = F.relu(self.actor_fc1(state))
        probs = self.head(self.actor_fc2(probs))
        
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        return value, probs
    
    def pi(self, s, a):
        s = torch.Tensor(s)
        #s = torch.unsqueeze(s, 0)
        _, probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    def get_action(self, state):
        state = torch.tensor(state)
        #state = torch.unsqueeze(state, 0)
        _, probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        
        action = probs.multinomial(num_samples=1)
        action = action.data
        
        action = action[0]
        return action
    
    def epsilon_greedy_action(self, state, epsilon = 0.1):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, 0)
        _, probs = self.forward(state)
        
        probs = torch.squeeze(probs, 0)
        
        if random.random() > epsilon:
            action = torch.tensor([torch.argmax(probs)])
        else:
            action = torch.rand(probs.shape).multinomial(num_samples=1)
        
        action = action.data
        action = action[0]
        return action
    
    def value(self, s):
        s = torch.tensor(s)
        s = torch.unsqueeze(s, 0)
        value, _ = self.forward(s)
        value = torch.squeeze(value, 0)
        value = value[0]
        
        return value    

    def update_weight(optimizer, states, actions, rewards, last_state, entropy_term = 0):
        # compute Q values
        Qval = model.value(last_state)
        loss = 0

        for s_t, a_t, r_tt in reversed(list(zip(states, actions, rewards))):
            log_prob = torch.log(model.pi(s_t, a_t))
            value = model.value(s_t)
            Qval = r_tt + GAMMA * torch.clone(Qval)

            advantage = Qval - value

            actor_loss = (-log_prob * advantage)
            critic_loss = 0.5 * advantage.pow(2)
            loss += actor_loss + critic_loss + 0.001 * entropy_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(env, model, optimizer, MAX_EPISODES, MAX_TIMESTEPS, learning_rate, step_size):
        try:
            returns = []

            for i_episode in range(MAX_EPISODES):

                #state = env.init_obs
                state = env.reset()
                init_state = state

                done = False

                states = []
                actions = []
                rewards = []   # no reward at t = 0

                #while not done:
                for timesteps in range(MAX_TIMESTEPS):

                    states.append(state)

                    action = model.get_action(state)
                    actions.append(action)

                    state, reward, done, _ = env.step(action.tolist())
                    rewards.append(reward)

                    if done or timesteps == MAX_TIMESTEPS-1:
                        last_state = state
                        break

                update_weight(optimizer, states, actions, rewards, last_state)

                if (i_episode + 1) % 500 == 0:
                    print("Episode {} return {}".format(i_episode + 1, sum(rewards)))
                    torch.save(model, './saved_models/model' + str(i_episode + 1) + '.pt')
                elif (i_episode + 1) % 10 == 0:
                    print("Episode {} return {}".format(i_episode + 1, sum(rewards)))

                returns.append(sum(rewards))

        except KeyboardInterrupt:
            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        env.close()

class ActorCritic(nn.Module):
    def __init__(self, **params) , inputs, outputs, env, model, optimizer, MAX_EPISODES, MAX_TIMESTEPS, learning_rate, step_size):
        super(ActorCritic, self).__init__()
        
        # for Actor
        self.actor_fc1 = nn.Linear(params[''], 256)
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.critic_fc2 = nn.Linear(256, 1)
        self.head = nn.Softmax(dim=0)
        
        # torch.log makes nan(not a number) error so we have to add some small number in log function
        self.ups=1e-7

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        state = x.to(device)
        
        probs = F.relu(self.actor_fc1(state))
        probs = self.head(self.actor_fc2(probs))
        
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        return value, probs
    
    def pi(self, s, a):
        s = torch.Tensor(s)
        #s = torch.unsqueeze(s, 0)
        _, probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    def get_action(self, state):
        state = torch.tensor(state)
        #state = torch.unsqueeze(state, 0)
        _, probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        
        action = probs.multinomial(num_samples=1)
        action = action.data
        
        action = action[0]
        return action
    
    def epsilon_greedy_action(self, state, epsilon = 0.1):
        state = torch.tensor(state)
        state = torch.unsqueeze(state, 0)
        _, probs = self.forward(state)
        
        probs = torch.squeeze(probs, 0)
        
        if random.random() > epsilon:
            action = torch.tensor([torch.argmax(probs)])
        else:
            action = torch.rand(probs.shape).multinomial(num_samples=1)
        
        action = action.data
        action = action[0]
        return action
    
    def value(self, s):
        s = torch.tensor(s)
        s = torch.unsqueeze(s, 0)
        value, _ = self.forward(s)
        value = torch.squeeze(value, 0)
        value = value[0]
        
        return value    

    def update_weight(optimizer, states, actions, rewards, last_state, entropy_term = 0):
        # compute Q values
        Qval = model.value(last_state)
        loss = 0

        for s_t, a_t, r_tt in reversed(list(zip(states, actions, rewards))):
            log_prob = torch.log(model.pi(s_t, a_t))
            value = model.value(s_t)
            Qval = r_tt + GAMMA * torch.clone(Qval)

            advantage = Qval - value

            actor_loss = (-log_prob * advantage)
            critic_loss = 0.5 * advantage.pow(2)
            loss += actor_loss + critic_loss + 0.001 * entropy_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(env, model, optimizer, MAX_EPISODES, MAX_TIMESTEPS, learning_rate, step_size):
        try:
            returns = []

            for i_episode in range(MAX_EPISODES):

                #state = env.init_obs
                state = env.reset()
                init_state = state

                done = False

                states = []
                actions = []
                rewards = []   # no reward at t = 0

                #while not done:
                for timesteps in range(MAX_TIMESTEPS):

                    states.append(state)

                    action = model.get_action(state)
                    actions.append(action)

                    state, reward, done, _ = env.step(action.tolist())
                    rewards.append(reward)

                    if done or timesteps == MAX_TIMESTEPS-1:
                        last_state = state
                        break

                update_weight(optimizer, states, actions, rewards, last_state)

                if (i_episode + 1) % 500 == 0:
                    print("Episode {} return {}".format(i_episode + 1, sum(rewards)))
                    torch.save(model, './saved_models/model' + str(i_episode + 1) + '.pt')
                elif (i_episode + 1) % 10 == 0:
                    print("Episode {} return {}".format(i_episode + 1, sum(rewards)))

                returns.append(sum(rewards))

        except KeyboardInterrupt:
            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        env.close()

MAX_EPISODES = 3000
MAX_TIMESTEPS = 1000

ALPHA = 3e-4 # learning rate
GAMMA = 0.99 # step-size

#env = MazeSolverEnv()
env = gym.make('CartPole-v0')
#num_actions = env.num_action
num_actions = env.action_space.n
#num_states = 365
num_states = env.observation_space.shape[0]

actor_critic = ActorCritic(num_states, num_actions).to(device)
optimizer = optim.Adam(actor_critic.parameters(), lr=ALPHA)

ActorCritic_paramters = {
    num_actions = num_actions, 
    num_states = num_states, 
    env = env, 
    model = actor_critic, 
    optimizer = optimizeGr, 
    MAX_EPISODES = MAX_EPISODES, 
    MAX_TIMESTEPS = MAX_TIMESTEPS, 
    learning_rate = ALPHA, 
    step_size = GAMMA
}

train(env, actor_critic, optimizer, MAX_EPISODES, MAX_TIMESTEPS, ALPHA, GAMMA)

