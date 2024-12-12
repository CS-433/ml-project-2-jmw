import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import csv
import Augment
import os
import random
from copy import deepcopy

from Config import Config
from Models import KeypointDetectionModel



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



def build_error_data(kpd_model, data, device = "mps", norm_min = None, norm_max = None):
    """
    Run the KPD model to get the error data. 
    Parameters:
    ----------
    kpd_model : the kpd_model
    data: as always a list of dict
    device: which device to run the kpd model on
    norm_min and norm_max: 
        we normalize the kpd_model errors using min max normalization because they are small. By default,
        norm_min = None, norm_max = None implies that the min and max will be computed. Different normalization 
        values can be given, 0 and 1 to avoid normalization.

    Returns:
    -------
    error_data: a list of dictionnaries with entries "Image Name" and "Error" (the mse of the kpd model on that image)
    norm_min, norm_max: the computed normalization min and max
    """
    print("Building error data for confidence model")
    with torch.no_grad():
        kpd_model.eval()
        images, keypoints, names = KeypointDetectionModel.get_full_unshuffled_batch(data, augment_images=False)
        images = images.to(device)
        keypoints = keypoints.to(device)
        outputs = kpd_model(images)
        errors = (keypoints - outputs)**2
        mse_per_vector = torch.mean(errors, dim=1, keepdim=True)
        if not(norm_min == 0 and norm_max == 1):
            norm_min, norm_max = torch.min(mse_per_vector), torch.max(mse_per_vector)
        
        mse_per_vector = (mse_per_vector - norm_min)/(norm_max - norm_min)
    
    #data_with_errors = deepcopy(data)

    #for i in range(len(data)):
    #    data_with_errors[i]["Error"] = mse_per_vector[i].item()

    error_data = [{"Image Name": names[i], "Error": mse_per_vector[i].item()} for i in range(len(names))]
    
    return error_data, norm_min, norm_max


def get_batch(data, kpd_model, batchsize = 25, augment_images = True, device = "mps"):

    """
    Generates a batch of data for the model.
    Note that this version always runs the KPD model's predictions.
    Therefore it is relatively safe from programmer mistakes (we are sure that the measured error)
    comes from the given image but it is super slow because it doesn't use pre-computed errors.

    Parameters:
    ----------
    data : list of dict
        A list where each dictionary represents an image and its keypoints:
        "Image Name", "x1", "y1", "x2", "y2".
    batchsize : int, optional
        The number of samples per batch (default is 25).
    augment_images : bool, optional
        Whether to apply image augmentation (default is True).
    device: which device to run the kpd predictions on

    Returns:
    -------
    tuple:
        A batch of images and corresponding labels/errors.
    """

    kpd_model.eval()

    features, targets = [], []
    dataset = []
    
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


def get_batch_fast(error_data, batchsize = 25, augment_images = False, device = "mps"):

    """
    Faster version of get_batch. Much faster because uses precomputed errors and doesn't run the 
    KPD model. However, a mistake could appear somewhere down the line that causes the errors to be unrelated 
    to the images (a batch getting shuffled somewhere).
    
    Compared to get_batch, data has been replaced by error_data which is still a list of dict, but with entries:
    "Image Name", "Error".
    """

    features, errors = [], []
    
    # Add the image to the image_data dictionnary (seemed more convenient but might actually be stupid)
    names = []
    for error_dict in random.sample(error_data, batchsize):
        name = error_dict["Image Name"]
        error = error_dict["Error"]
        img_path = os.path.join(Config.images_folder_path + "/", name)
        # Replace the suffix with .png
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}.png"
        img, keypoints = Augment.prepare_for_model(img_path, [], augment_images=augment_images)

        x, y = KeypointDetectionModel.to_xy(img, keypoints)
        features.append(x.unsqueeze(0))
        errors.append(torch.tensor([error]))
        names.append(name)

    features_tensor, error_tensor = torch.stack(features).to(device), torch.stack(errors).to(device)

    return features_tensor, error_tensor, names


"""
We want to normalize the KPD Model error to make training more reliable. So the following function aims to estimates 
max and min values of the error.
"""
def get_error_normalization(kpd_model, data, batchsize = 100, augment_images = False, device = "mps"):
    kpd_model.eval()
    with torch.no_grad():
        images, errors, names = get_batch(data, kpd_model, batchsize=batchsize, augment_images=augment_images, device=device)
    
    min_error, max_error = torch.min(errors), torch.max(errors)
    normalizer = lambda error: (error-min_error)/(max_error-min_error)
    print("Computed error-normalization function.")
    return normalizer



def train_conf_model(conf_model, kpd_model, train_data, test_data, batchsize, test_batchsize, epochs, initial_lr = 1e-5, lr_decay = 0.99, device = "mps", augment_training_images = False, feedback_rate = 20, normalize_errors = True):
    
    """
    Train our confidence model.

    Parameters:
    ----------
    conf_model: the confidence model.
    kpd_model : torch.nn.Module
        The keypoint detection model.
    train_data : Any
        The training dataset, expected to work with `model.get_batch` for batch retrieval.
        Is in the form of a list of dictionnaries. Each dict is a datapoint and has entries "Image Name", "x1", "y1", "x2", "y2".
    test_data : Any
        The testing/validation dataset, same form as train_data.
    batchsize : int
        The number of training samples in each batch.
    test_batchsize : int
        The number of testing samples in each test batch.
    epochs : int
        The total number of epochs for training.
    initial_lr : float, optional
        The initial learning rate for the optimizer (default: 1e-5).
    lr_decay : float, optional
        Multiplicative factor for learning rate decay at each epoch (default: 0.99).
    device : str, optional
        The device to run training on (e.g., "cpu", "cuda", "mps") (default: "mps").
    augment_training_images : bool, optional
        Whether to apply image augmentation to the training data (default: False).
        Results seem better without augmentation: trade-off more specific model vs more robust.
    feedback_rate : int, optional
        Interval (in epochs) at which feedback (e.g., test loss and predictions) is provided (default: 50).
    normalize_errors: bool, optional
        Somehow we get better results without normalization. Needs investigation.
    """

    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(conf_model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    conf_model.train()
    kpd_model.eval()

    # Note: might not be great conceptually to use the test data to find the normalization.
    if normalize_errors:
        normalization_function = get_error_normalization(kpd_model, train_data, batchsize=400, augment_images=augment_training_images, device=device)
    else: normalization_function = lambda x: x

    for epoch in range(epochs):
        images, errors, names = get_batch(train_data, kpd_model, batchsize=batchsize, augment_images=augment_training_images, device = device) 
        images = images.to(device)
        errors = errors.to(device)
        errors = normalization_function(errors) 

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
                errors = normalization_function(errors)
                images = images.to(device)
                errors = errors.to(device)
                errors = normalization_function(errors)
                #print(torch.max(errors))
                #print(torch.min(errors))

                #print(outputs, errors)
                outputs = conf_model(images)  # Forward pass
                test_loss = criterion(outputs, errors)  # Compute loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: test loss = {test_loss}, lr = {current_lr}")

                conf_model.train()
        
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = conf_model.state_dict()

    conf_model.load_state_dict(best_model_state)



"""
Fast version of train_conf_model. Uses the pre-computed errors (list of dicts with entries "Image Name" and "Error".)
"""
def train_conf_model_fast(conf_model, train_error_data, test_error_data, batchsize, test_batchsize, epochs, initial_lr = 1e-5, lr_decay = 0.99, device = "mps", augment_training_images = False, feedback_rate = 20, normalize_errors = True):
    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(conf_model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    conf_model.train()

    # Note: might not be great conceptually to use the test data to find the normalization.

    for epoch in range(epochs):
        images, errors, names = get_batch_fast(train_error_data, batchsize=batchsize, augment_images=augment_training_images, device = device) 
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
                images, errors, names = get_batch_fast(test_error_data, batchsize = test_batchsize, augment_images = False, device = device)  
                images = images.to(device)
                errors = errors.to(device)
                #print(torch.max(errors))
                #print(torch.min(errors))

                #print(outputs, errors)
                outputs = conf_model(images)  # Forward pass
                test_loss = criterion(outputs, errors)  # Compute loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: test loss = {test_loss}, lr = {current_lr}")

                conf_model.train()
        
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = conf_model.state_dict()

    conf_model.load_state_dict(best_model_state)

