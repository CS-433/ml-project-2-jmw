import torch
import torch.nn as nn
import torch.nn.functional as F


from Config import Config



"""
The confidence model aims to learn from the images and the KPD Models predictions which images are "hard".
For example, an unusual body shape or keypoints hidden by the ant's legs make it hard for the model to find 
the Keypoints. The aim is to be able to automatically make measurements on the "easy" images, while "hard" images
will still require human measures.

It learns to predict the MSE of the trained KPD model on images.
"""



"""
Simple architecture with convolutionnal layers followed by fully connected layers.
"""
class ConfidenceModel(nn.Module):
    def __init__(self):
        input_shape = Config.input_image_shape
        super(ConfidenceModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves dimensions
        
        self.flatten = torch.nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * int(input_shape[0]/4) * int(input_shape[1]/4), 512)  # Input: Flattened pooled output
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: 2 estimations of the error (one for each keypoint)
    

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv3(x))) 
        #print(x.shape)
        
        # Flatten for fully connected layers
        #x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64 * 25 * 50)
        x = self.flatten(x)  # Flatten before FC layer
        #print(x.shape)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final output: (batch_size, 1)
        #print(x.shape)
        # Sigmoid or id / relu?
        #x = F.relu(self.fc3(x))  # Final output: (batch_size, 1)
        
        return x


