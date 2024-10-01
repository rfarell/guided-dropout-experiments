import torch
import torch.nn as nn
import torch.nn.functional as F

class NoDropoutModel(nn.Module):
    def __init__(self, num_classes=10):
        super(NoDropoutModel, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 8192)
        self.fc2 = nn.Linear(8192, 8192)
        self.fc3 = nn.Linear(8192, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
