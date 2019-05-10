
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from random import *


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #Seed
        #i = 0
        #random_state = 42
        #pseudo_numbers = []
        #for i in range(6):
        #	pseudo_numbers.append(random())
        #print("The pseudo number list is:", pseudo_numbers)


        #Convolutional layers (https://pytorch.org/docs/stable/nn.html#id21)
        self.conv01 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 7, bias = True)
        self.conv02 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, bias = True)
        self.conv03 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, bias = True)
        self.conv04 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2, bias = True)
        
        #MaxPool layer (https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d)
        self.pool = nn.MaxPool2d(kernel_size = 5, stride = 3)

        #Fully Conected Layer #36864#
        self.fc01 = nn.Linear(in_features = 256, out_features = 136) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        #self.fc02 = nn.Linear(in_features = 500,    out_features = 200)
        #self.fc03 = nn.Linear(in_features = 200,    out_features = 136) # The output is 136 points [68 keypoint (x, y)]

        #Adding Dropout Layers (https://pytorch.org/docs/stable/nn.html#dropout-layers)
#         self.drop02 = nn.Dropout(p = pseudo_numbers[0])
#         self.drop02 = nn.Dropout(p = pseudo_numbers[1])
#         self.drop03 = nn.Dropout(p = pseudo_numbers[2])
#         self.drop04 = nn.Dropout(p = pseudo_numbers[3])

#         #dropout for fully conected layers
#         self.drop05 = nn.Dropout(p = pseudo_numbers[4])
#         self.drop06 = nn.Dropout(p = pseudo_numbers[5])
        self.drop02 = nn.Dropout(0.3)
        self.drop02 = nn.Dropout(0.4)
        self.drop03 = nn.Dropout(0.45)
        self.drop04 = nn.Dropout(0.45)

        #dropout for fully conected layers
        self.drop05 = nn.Dropout(0.6)
        self.drop06 = nn.Dropout(0.7)


        
    def forward(self, x):

        #Convolution + Activation + Pooling + Dropout

        x = self.pool(F.relu(self.conv01(x)))
        #print("First size: ", x.shape)

        x = self.drop02(self.pool(F.relu(self.conv02(x))))
        #print("Second size: ", x.shape)

        x = self.drop03(self.pool(F.relu(self.conv03(x))))
        #print("Third size: ", x.shape)

        x = self.drop04(self.pool(F.relu(self.conv04(x))))
        #print("Forth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        #x = self.drop05(F.relu(self.fc01(x)))
        #print("First dense size: ", x.shape)

        #x = self.drop06(F.relu(self.fc02(x)))
        #print("Second dense size: ", x.shape)

        # Final Layer
        x = self.fc01(x)
        #print("Final dense size: ", x.shape)
        
        
        
        return x