# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_V1(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN_V1, self).__init__()
        self.actor_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.actor_bn1 = nn.BatchNorm2d(16)
        self.actor_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.actor_bn2 = nn.BatchNorm2d(32)
        self.actor_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.actor_bn3 = nn.BatchNorm2d(32)
        self.actor_bn4 = nn.BatchNorm1d(4) # 4 actions
        self.actor_tanh = nn.Tanh()
        
        self.critic_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.critic_bn1 = nn.BatchNorm2d(16)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.critic_bn2 = nn.BatchNorm2d(32)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.critic_bn3 = nn.BatchNorm2d(32)
        self.critic_bn4 = nn.BatchNorm1d(1)
        self.critic_tanh = nn.Tanh()
        
        # torch.log makes nan(not a number) error so we have to add some small number in log function
        self.ups=1e-7

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.actor_fc1 = nn.Linear(linear_input_size, outputs)
        self.head = nn.Softmax(dim=1)
        
        self.critic_fc1 = nn.Linear(linear_input_size, 1)

    def forward(self, x):
        state = x
        state = torch.unsqueeze(state, 0)
        
        probs = F.relu(self.actor_conv1(state))
        probs = F.relu(self.actor_conv2(probs))
        probs = F.relu(self.actor_conv3(probs))
        probs = torch.flatten(probs, 1)
        probs = self.head(self.actor_fc1(probs))
        
        value = F.relu(self.critic_conv1(state))
        value = F.relu(self.critic_conv2(value))
        value = F.relu(self.critic_conv3(value))
        value = torch.flatten(value, 1)
        value = F.relu(self.critic_fc1(value))
        
        return value, probs
