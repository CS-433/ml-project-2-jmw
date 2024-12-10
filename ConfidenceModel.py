import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import csv
import Augment
import os
import random

from Config import Config
import KeypointDetectionModel



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
        # Sigmoid or id ?
        #x = F.sigmoid(self.fc3(x))  # Final output: (batch_size, 4)
        
        return x


def get_batch(data, kpd_model, batchsize = 25, augment_images = True, device = "mps"):

    kpd_model.eval()

    features, targets = [], []
    dataset = []

    """
    # Open the CSV file and read its contents into a dictionary
    # The stored data is: picture_name, x1, y1, x2, y2 
    with open(Config.coords_file_path, mode='r') as file:
        reader = csv.DictReader(file)  # Use DictReader to automatically map rows to dictionaries
        data = [row for row in reader]  # Convert each row into a dictionary and store in a list
    """
    
    # Add the image to the image_data dictionnary (seemed more convenient but might actually be stupid)
    names = []
    for image_data in random.sample(data, batchsize):
        #print(image_data)
        name = image_data["Image Name"]
        img_path = os.path.join(Config.images_folder_path + "/", name)
        # Replace the suffix with .png
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}.png"
        x1, y1 = image_data["x1"], image_data["y1"]
        x2, y2 = image_data["x2"], image_data["y2"]
        #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        img, keypoints = Augment.prepare_for_model(img_path, [(x1, y1), (x2, y2)], augment_images=augment_images)

        if len(keypoints) == 2:
            x, y = KeypointDetectionModel.to_xy(img, keypoints)
            features.append(x.unsqueeze(0))
            targets.append(y)
            dataset.append((x, y))
            names.append(name)
    
    features_tensor, targets_tensor = torch.stack(features).to(device), torch.stack(targets).to(device)
    kpd_model.eval()
    kpd_pred = kpd_model(features_tensor)

    # Compute the element-wise difference
    difference = kpd_pred - targets_tensor

    # Square the differences
    squared_difference = difference ** 2

    # Compute the mean along the last dimension (the vector components)
    mse_per_vector = torch.mean(squared_difference, dim=1, keepdim=True)

    return features_tensor, mse_per_vector, names


def train_conf_model(conf_model, kpd_model, train_data, test_data, batchsize, test_batchsize, epochs, initial_lr = 1e-5, lr_decay = 0.99, device = "mps", augment_training_images = False, feedback_rate = 20):
    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(conf_model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    conf_model.train()
    kpd_model.eval()

    for epoch in range(epochs):
        images, errors, names = get_batch(train_data, kpd_model, batchsize=batchsize, augment_images=augment_training_images, device = device)  
        images = images.to(device)
        errors = errors.to(device)

        optimizer.zero_grad()
        outputs = conf_model(images)  # Forward pass
        loss = criterion(outputs, errors)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        scheduler.step() # Update LR 

        if not epoch % feedback_rate:
            with torch.no_grad():
                conf_model.eval()
                images, errors, names = get_batch(test_data, kpd_model, batchsize = test_batchsize, augment_images = False, device = device)  
                images = images.to(device)
                errors = errors.to(device)

                outputs = conf_model(images)  # Forward pass
                test_loss = criterion(outputs, errors)  # Compute loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: test loss = {test_loss}, lr = {current_lr}")

                conf_model.train()
        
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = conf_model.state_dict()

    conf_model.load_state_dict(best_model_state)


