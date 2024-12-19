import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config

import Augment

"""
In this file we define and train a KeyPoint Detection (KPD) Model from scratch.
"""


"""
Simple architecture with convolutionnal layers followed by fully connected layers.
"""


class KeypointDetectionModel(nn.Module):
    def __init__(self, input_image_shape=Config.input_image_shape_basic_model):
        self.input_shape = input_image_shape
        self.pipeline, self.augment_pipeline = Augment.create_transform_pipeline(
            False, self.input_shape
        ), Augment.create_transform_pipeline(True, self.input_shape)

        super(KeypointDetectionModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * int(self.input_shape[0] / 4) * int(self.input_shape[1] / 4), 512
        )  # Input: Flattened pooled output
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # Output: 4 coordinates [x1, y1, x2, y2]

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)

        # Flatten for fully connected layers
        # x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64 * 25 * 50)
        x = self.flatten(x)  # Flatten before FC layer
        # print(x.shape)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final output: (batch_size, 4)
        # print(x.shape)

        return x
