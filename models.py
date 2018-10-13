## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)

        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(46080, 1000)
        self.fc2 = nn.Linear(1000, 136)
        

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        print ("input: ", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        print ("after conv1", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        print ("after conv2", x.size())

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        print ("after flatten ", x.size())
        
        x = F.relu(self.fc1(x))
        print ("after 1st lineal ", x.size())
        
        x = F.relu(self.fc2(x))
        print ("after 2nd lineal ", x.size())

        #x = F.log_softmax(x, dim=1)
        
        # final output
        return x

# instantiate and print your Net
#net = Net()
#print(net)