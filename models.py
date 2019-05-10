import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Covolutional
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)

        # Maxpooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected
        self.fc1 = nn.Linear(36864,1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)

    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool(F.relu(self.conv2(x))))

        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop3(self.pool(F.relu(self.conv3(x))))

        # Forth - Convolution + Activation + Pooling + Dropout
        x = self.drop4(self.pool(F.relu(self.conv4(x))))

        # Flattening the layer
        x = x.view(x.size(0), -1)

        # First - Dense + Activation + Dropout
        x = self.drop5(F.relu(self.fc1(x)))

        # Second - Dense + Activation + Dropout
        x = self.drop6(F.relu(self.fc2(x)))

        # Final Dense Layer
        x = self.fc3(x)

        return x