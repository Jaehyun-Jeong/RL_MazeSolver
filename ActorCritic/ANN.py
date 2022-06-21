
import numpy as np
import random
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Environment 
import gym

'''
    ActorCritic_paramters = {
        device = device, # device to use, 'cuda' or 'cpu'
        env = env, # environment like gym
        model = actor_critic, # torch models for policy and value funciton
        optimizer = optimizeGr, # torch optimizer
        #MAX_EPISODES = MAX_EPISODES, # maximum episodes you want to learn
        maxTimesteps= MAX_TIMESTEPS, # maximum timesteps agent take 
        stepsize = GAMMA # step-size for updating Q value
    }
'''

class model(nn.Module):
    def __init__(self, inputs, outputs):
        super(model, self).__init__()

        # for Actor
        self.actor_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        state = x
        
        probs = F.relu(self.actor_fc1(state))
        probs = self.head(self.actor_fc2(probs))
        
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        return value, probs
        
class ActorCritic():
    def __init__(self, **params_dict): # parmas = {env, model, optimizer, maxTimesteps, stepsize}
        super(ActorCritic, self).__init__()

        # init parameters 
        self.device = params_dict['device']
        self.env = params_dict['env']
        self.model = params_dict['model']
        self.optimizer = params_dict['optimizer']
        self.maxTimesteps = params_dict['maxTimesteps'] 
        self.stepsize = params_dict['stepsize']
        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    def get_action(self, s):
        s = torch.tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        
        a = probs.multinomial(num_samples=1)
        a = a.data
        
        action = a[0]
        return action
    
    def epsilon_greedy_action(self, s, epsilon = 0.1):
        s = torch.tensor(s).to(self.device)
        s = torch.unsqueeze(s, 0)
        _, probs = self.model.forward(s)
        
        probs = torch.squeeze(probs, 0)
        
        if random.random() > epsilon:
            a = torch.tensor([torch.argmax(probs)])
        else:
            a = torch.rand(probs.shape).multinomial(num_samples=1)
        
        a = a.data
        action = a[0]
        return action
    
    def value(self, s):
        s = torch.tensor(s).to(self.device)
        s = torch.unsqueeze(s, 0)
        value, _ = self.model.forward(s)
        value = torch.squeeze(value, 0)
        value = value[0]
        
        return value    

    def update_weight(self, states, actions, rewards, last_state, entropy_term = 0):
        # compute Q values
        Qval = self.value(last_state)
        loss = 0

        for s_t, a_t, r_tt in reversed(list(zip(states, actions, rewards))):
            log_prob = torch.log(self.pi(s_t, a_t))
            value = self.value(s_t)
            Qval = r_tt + self.stepsize * torch.clone(Qval)
            advantage = Qval - value

            actor_loss = (-log_prob * advantage)
            critic_loss = 0.5 * advantage.pow(2)
            loss += actor_loss + critic_loss + 0.001 * entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, maxEpisodes):
        try:
            returns = []

            for i_episode in range(maxEpisodes):

                state = env.reset()
                init_state = state

                done = False

                states = []
                actions = []
                rewards = []   # no reward at t = 0

                #while not done:
                for timesteps in range(self.maxTimesteps):

                    states.append(state)

                    action = self.get_action(state)
                    actions.append(action)

                    state, reward, done, _ = self.env.step(action.tolist())
                    rewards.append(reward)

                    if done or timesteps == self.maxTimesteps-1:
                        last_state = state
                        break

                self.update_weight(states, actions, rewards, last_state)

                returns.append(sum(rewards))

                if (i_episode + 1) % 500 == 0:
                    print("Episode {} return {}".format(i_episode + 1, returns[-1]))

                    # SAVE THE MODEL
                    #torch.save(model, '../saved_models/model' + str(i_episode + 1) + '.pt')

                elif (i_episode + 1) % 10 == 0:
                    print("Episode {} return {}".format(i_episode + 1, returns[-1]))

        except KeyboardInterrupt:
            print("==============================================")
            print("KEYBOARD INTERRUPTION!!=======================")
            print("==============================================")
            #plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        env.close()

if __name__ == "__main__":
    MAX_EPISODES = 3000
    MAX_TIMESTEPS = 1000

    ALPHA = 3e-4 # learning rate
    GAMMA = 0.99 # step-size

    # device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set environment
    env = gym.make('CartPole-v0')

    # set ActorCritic
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    ACmodel = model(num_states, num_actions).to(device)
    optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

    ActorCritic_parameters = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        #MAX_EPISODES = MAX_EPISODES, # maximum episodes you want to learn
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'stepsize': GAMMA # step-size for updating Q value
    }

    # Initialize Actor-Critic Mehtod
    AC = ActorCritic(**ActorCritic_parameters)

    # TRAIN Agent
    AC.train(MAX_EPISODES)

