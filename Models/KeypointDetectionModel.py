import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import csv
import Augment
import os
import random

import imageProcessing as ip
from matplotlib import pyplot as plt

from Config import Config


# Transform an image and keypoints = [(x1, y1), (x2, y2)] into x and y for the model. 
# The image should already have the right shape for the model.
def to_xy(image, key_points = []):
    shape = Config.input_image_shape
    if image.shape != shape:
        raise ValueError("Image shape doesn't match config shape !")
    image = image/np.max(image) #normalize
    if len(key_points) == 2:
        x1, y1, x2, y2 = key_points[0][0], key_points[0][1], key_points[1][0], key_points[1][1]
        x1, x2 = x1/shape[1], x2/shape[1]
        y1, y2 = y1/shape[0], y2/shape[0]

        return torch.from_numpy(image).float(), torch.asarray([x1, x2, y1, y2])

    return torch.from_numpy(image).float(), []




class KeypointDetectionModel(nn.Module):
    def __init__(self):
        input_shape = Config.input_image_shape
        super(KeypointDetectionModel, self).__init__()
        
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
        self.fc3 = nn.Linear(128, 4)  # Output: 4 coordinates [x1, y1, x2, y2]
    

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
        x = self.fc3(x)  # Final output: (batch_size, 4)
        #print(x.shape)
        
        return x


def get_batch(data, batchsize = 25, augment_images = True):

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
            x, y = to_xy(img, keypoints)
            features.append(x.unsqueeze(0))
            targets.append(y)
            dataset.append((x, y))
            names.append(name)
    
    features_tensor, targets_tensor = torch.stack(features), torch.stack(targets)
    return features_tensor, targets_tensor, names


def get_full_unshuffled_batch(data, augment_images = True):

    features, targets = [], []
    dataset = []
    
    # Add the image to the image_data dictionnary (seemed more convenient but might actually be stupid)
    names = []
    for image_data in data:
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
            x, y = to_xy(img, keypoints)
            features.append(x.unsqueeze(0))
            targets.append(y)
            dataset.append((x, y))
            names.append(name)
    
    features_tensor, targets_tensor = torch.stack(features), torch.stack(targets)
    return features_tensor, targets_tensor, names



"""
Plot keypoint detection model prediction side by side with expected keypoints
"""
def plot_model_prediction(kpd_model, data, n_images, augment_images = False, device=None, conf_model = None, error_estimation_interval = [-1, 1]):

    # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    kpd_model.eval()

    # Sample n_images images to plot
    for image_data in random.sample(data, n_images):

        # Get image path. All images without bachground for the model are .png, so we need to change the suffix
        name = image_data["Image Name"]
        img_path = os.path.join(Config.images_folder_path + "/", name)
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}.png"

        # Get keypoints
        x1, y1 = image_data["x1"], image_data["y1"]
        x2, y2 = image_data["x2"], image_data["y2"]

        # Prepare image and keypoints for the model
        img, keypoints = Augment.prepare_for_model(img_path, [(x1, y1), (x2, y2)], augment_images=augment_images)
        x, _ = to_xy(img, keypoints)
        # We somehow have to unsqueeze twice, could be worth investigating
        x_tensor = x.unsqueeze(0).unsqueeze(0).to(device)  # Move to the correct device (MPS or CPU)
        
        # Forward pass on the model
        with torch.no_grad():  # No gradients needed for inference
            pred = kpd_model(x_tensor)

        # Post-processing of the model output (e.g., rescale the coords which are outputed between 0 and 1)
        """
        This is completely messed up but somehow we have to switch x and y (it should be x1, y1, x2, y2 = ...)
        Somewhere in the code x and y were already switched, we need to find where...
        """
        y1pred, x1pred, y2pred, x2pred = pred[0][0].item() * Config.input_image_shape[0], pred[0][1].item() * Config.input_image_shape[1], pred[0][2].item() * Config.input_image_shape[0], pred[0][3].item() * Config.input_image_shape[1]

        #img, keypoints = Augment.prepare_for_model(img_path, [(x1, y1), (x2, y2)])
        
        # If len(keypoints) != 2, then the augmentation lost a keypoint (too much rotation for example) and we are not interested
        if len(keypoints) == 2:

            error_pred = None

            if conf_model is None:
                plot = True
            else:
                conf_model.eval()
                with torch.no_grad():
                    error_pred = conf_model(x_tensor)
                    min_error, max_error = error_estimation_interval[0], error_estimation_interval[1]
                    plot = True if (min_error <= error_pred and error_pred <= max_error) else False
            
            if plot:
                if not error_pred is None:
                    print("Error estimation = ", error_pred.item())

                # Add keypoints and prediction to the image and plot
                with_y = ip.add_point_channels(img, keypoints[0], keypoints[1])
                with_pred = ip.add_point_channels(img, (int(x1pred), int(y1pred)), (int(x2pred), int(y2pred)))

                # Plot side-by-side
                plt.figure(figsize=(10, 5))  # Set the figure size

                # First image
                plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
                plt.imshow(with_y)  # Display the first image
                plt.title("Expected")
                plt.axis('off')  # Turn off axes for better visual effect

                # Second image
                plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
                plt.imshow(with_pred)  # Display the second image
                plt.title("Prediction")
                plt.axis('off')

                # Display the plot
                plt.tight_layout()  # Adjust spacing
                plt.show()



def train_kpd_model(kpd_model, train_data, test_data, batchsize, test_batchsize, epochs, initial_lr = 1e-4, lr_decay = 0.99, device = "mps", augment_training_images = False, feedback_rate = 50, plot_predictions = False):
    
    """
    Train our keypoint detection (KPD) model using Mean Squared Error (MSE) loss and an Adam optimizer.

    Parameters:
    ----------
    kpd_model : torch.nn.Module
        The keypoint detection model to be trained.
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
        The initial learning rate for the optimizer (default: 1e-4).
    lr_decay : float, optional
        Multiplicative factor for learning rate decay at each epoch (default: 0.99).
    device : str, optional
        The device to run training on (e.g., "cpu", "cuda", "mps") (default: "mps").
    augment_training_images : bool, optional
        Whether to apply image augmentation to the training data (default: False).
        Results seem better without augmentation: trade-off more specific model vs more robust.
    feedback_rate : int, optional
        Interval (in epochs) at which feedback (e.g., test loss and predictions) is provided (default: 50).
    plot_predictions : bool, optional
        Whether to visualize model predictions during feedback steps (default: False).

    Returns:
    -------
    None

    Updates:
    -------
    - Trains the model by iterating over training data for the specified number of epochs.
    - Tracks and prints the test loss and learning rate at regular intervals.
    - Optionally plots predictions during feedback steps.
    - Loads the state of the model corresponding to the lowest test loss.


    Example:
    --------
    train_kpd_model(
        kpd_model=my_model, 
        train_data=train_dataset, 
        test_data=test_dataset, 
        batchsize=50, 
        test_batchsize=100, 
        epochs=200, 
        initial_lr=1e-4, 
        lr_decay=0.98, 
        device="mps", 
        augment_training_images=True, 
        feedback_rate=20, 
        plot_predictions=True
    )
    """

    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(kpd_model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    for epoch in range(epochs):
        images, keypoints, names = get_batch(train_data, batchsize=batchsize, augment_images=augment_training_images)  
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = kpd_model(images)  # Forward pass
        loss = criterion(outputs, keypoints)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        scheduler.step() # Update LR 

        if not epoch%feedback_rate:
            with torch.no_grad():
                kpd_model.eval()
                images, keypoints, names = get_batch(test_data, batchsize=test_batchsize, augment_images = False)  
                images = images.to(device)
                keypoints = keypoints.to(device)

                outputs = kpd_model(images)  # Forward pass
                test_loss = criterion(outputs, keypoints)  # Compute loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: test loss = {test_loss}, lr = {current_lr}")

                if plot_predictions:
                    plot_model_prediction(kpd_model, test_data, 2, device="mps", augment_images=False)
                kpd_model.train()
        
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = kpd_model.state_dict()
    
    kpd_model.load_state_dict(best_model_state)
    



