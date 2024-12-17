import torch
import torch.nn as nn

import random
import os
from Config import Config
import Augment
import numpy as np
from matplotlib import pyplot as plt
import imageProcessing as ip

import torch.nn.functional as F




def to_xy(image, key_points = []):
    """
    Transforms and normalizes an image and its corresponding keypoints into the input 
    and desired output format for a model.

    Parameters:
    ----------
    image : numpy.ndarray
        A one-channel image represented as a numpy array. The shape of the 
        array must match the shape defined in `Config.input_image_shape`.
    key_points : list of tuples, optional
        A list containing two keypoints, where each keypoint is represented 
        as a tuple of (x, y) coordinates. Example: [(x1, y1), (x2, y2)].

    Returns:
    -------
    tuple:
        - torch.Tensor: The normalized image as a PyTorch tensor with type float.
        - torch.Tensor or list: If `key_points` contains two keypoints, returns a 
          PyTorch tensor with the normalized coordinates [x1, x2, y1, y2]. 
          Otherwise, returns an empty list.

    Raises:
    ------
    ValueError:
        If the shape of the input image does not match the expected shape 
        defined in `Config.input_image_shape`.

    Notes:
    ------
    - The image is normalized by dividing each pixel value by the maximum pixel 
      value in the image.
    - The keypoints are normalized based on the dimensions of the image.
    """

    shape = image.shape
    image = image/np.max(image) #normalize pixel activations
    if len(key_points) == 2:
        x1, y1, x2, y2 = key_points[0][0], key_points[0][1], key_points[1][0], key_points[1][1]
        x1, x2 = x1/shape[1], x2/shape[1] # Normalize coords
        y1, y2 = y1/shape[0], y2/shape[0] # Normalize coords

        return torch.from_numpy(image).float(), torch.asarray([x1, x2, y1, y2])

    return torch.from_numpy(image).float(), []




def get_batch(data, pipeline, batchsize = 25):

    """
    Generates a batch of data for the model.

    Parameters:
    ----------
    data : list of dict
        A list where each dictionary represents an image and its keypoints:
        "Image Name", "x1", "y1", "x2", "y2".
    batchsize : int, optional
        The number of samples per batch (default is 25).
    augment_images : bool, optional
        Whether to apply image augmentation (default is True).

    Returns:
    -------
    tuple:
        A batch of images and corresponding labels/coordinates.
    """

    features, targets = [], []
    dataset = []
    
    # Add the image to the image_data dictionnary (seemed more convenient but might actually be stupid)
    names = []
    for image_data in random.sample(data, batchsize):
        name = image_data["Image Name"]
        img_path = os.path.join(Config.images_folder_path + "/", name)
        # Replace the suffix with .png
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}.png"
        x1, y1 = image_data["x1"], image_data["y1"]
        x2, y2 = image_data["x2"], image_data["y2"]

        img, keypoints = Augment.prepare_for_model(img_path, pipeline, [(x1, y1), (x2, y2)])

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
Arguments:
    - kpd_model: the keypoint detection model
    - data: still a list of dict
    - n_images: how many examples to plot
    - augment_images: wether to augment the images
    - device: on which device (cpu, mps, cuda) to run the model
    - conf_model: optionnal, print the confidence model's expected error
    - error_estimation_interval: Only print images for which the confidence model's error estimate lies in this interval.
"""
def plot_model_prediction(kpd_model, data, n_images, augment_images = False, device="mps", conf_model = None, error_estimation_interval = [-1, 1], radius = 4):

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
        pipeline = kpd_model.pipeline if not augment_images else kpd_model.augment_pipeline
        img, keypoints = Augment.prepare_for_model(img_path, pipeline, [(x1, y1), (x2, y2)])
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
        y1pred, x1pred, y2pred, x2pred = pred[0][0].item() * kpd_model.input_shape[0], pred[0][1].item() * kpd_model.input_shape[1], pred[0][2].item() * kpd_model.input_shape[0], pred[0][3].item() * kpd_model.input_shape[1]

        
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
                with_y = ip.add_point_channels(img, keypoints[0], keypoints[1], radius = radius)
                with_pred = ip.add_point_channels(img, (int(x1pred), int(y1pred)), (int(x2pred), int(y2pred)), radius = radius)

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



def plot_losses(train_loss, test_loss, first_fraction_to_skip = 0):
    # train_loss and test_loss are lists of losses recorded at each epoch. To skip first part where loss is very high to get more details, set first_fraction_to_skip > 0 
    n_epochs = len(train_loss)
    plot_from_epoch = int(first_fraction_to_skip*n_epochs)
    epochs = range(plot_from_epoch, n_epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss[plot_from_epoch:], label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, test_loss[plot_from_epoch:], label='Test Loss', color='red', marker='x')
    
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to train the KPD model
def train_kpd_model(kpd_model, train_data, test_data, batchsize, num_epochs, feedback_rate = 10, device="mps", initial_lr = 1e-4, lr_decay = 0.99, augment_training_images = False, plot_progression = True):

    train_losses, test_losses = [], []

    kpd_model.to(device)
    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(kpd_model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    for epoch in range(num_epochs):
        kpd_model.train()

        #images, keypoints, names = get_batch(train_data, batchsize=batchsize, augment_images=augment_training_images)
        pipeline = kpd_model.augment_pipeline if augment_training_images else kpd_model.pipeline
        images, keypoints, names = get_batch(train_data, pipeline, batchsize=batchsize)
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
                #images, keypoints, names = get_batch(test_data, batchsize=len(test_data), augment_images = False)
                images, keypoints, names = get_batch(test_data, kpd_model.pipeline, batchsize=len(test_data))  
                images = images.to(device)
                keypoints = keypoints.to(device)

                outputs = kpd_model(images)  # Forward pass
                test_loss = criterion(outputs, keypoints)  # Compute loss
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: test loss = {test_loss}, lr = {current_lr}")
                print(f"Distance error = ", get_distance_loss(outputs, keypoints))

                """
                if plot_predictions:
                    plot_model_prediction(kpd_model, test_data, 2, device="mps", augment_images=False)
                """
                kpd_model.train()

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_state = kpd_model.state_dict()
        
        test_losses.append(test_loss.item())
        train_losses.append(loss.item())

    print("Training complete, reverting to best model state")
    kpd_model.load_state_dict(best_state)

    if plot_progression:
        plot_losses(train_losses, test_losses)
        plot_losses(train_losses, test_losses, first_fraction_to_skip=1/3)


def get_distance_loss(outputs, keypoints):
    """
    Computes the distance error (loss) between the predicted and ground-truth keypoints.

    Args:
        outputs (torch.Tensor): Predicted keypoints in the format [x1, y1, x2, y2].
        keypoints (torch.Tensor): Ground-truth keypoints in the format [x1, y1, x2, y2].

    Returns:
        torch.Tensor: The distance error as a scalar tensor.
    """
    # Ensure inputs are tensors
    outputs = outputs.view(-1, 4)  # Reshape to [batch_size, 4] if necessary
    keypoints = keypoints.view(-1, 4)

    # Extract predicted and ground-truth coordinates
    x1_pred, y1_pred, x2_pred, y2_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    x1_true, y1_true, x2_true, y2_true = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3]

    # Compute predicted and ground-truth distances
    pred_distances = torch.sqrt((x2_pred - x1_pred)**2 + (y2_pred - y1_pred)**2)
    true_distances = torch.sqrt((x2_true - x1_true)**2 + (y2_true - y1_true)**2)

    # Compute the absolute error between the distances
    distance_loss = F.l1_loss(pred_distances, true_distances)

    return distance_loss.item()


def get_full_unshuffled_batch(data, pipeline):

    """
    Same as get_batch above but to generate error data for the Confidence model. Doesn't shuffle the data and
    returns the full batch.
    """

    features, targets = [], []
    dataset = []
    
    # Add the image to the image_data dictionnary (seemed more convenient but might actually be stupid)
    names = []
    for image_data in data:
        name = image_data["Image Name"]
        img_path = os.path.join(Config.images_folder_path + "/", name)
        # Replace the suffix with .png
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}.png"
        x1, y1 = image_data["x1"], image_data["y1"]
        x2, y2 = image_data["x2"], image_data["y2"]

        img, keypoints = Augment.prepare_for_model(img_path, pipeline, [(x1, y1), (x2, y2)])

        if len(keypoints) == 2:
            x, y = to_xy(img, keypoints)
            features.append(x.unsqueeze(0))
            targets.append(y)
            dataset.append((x, y))
            names.append(name)
    
    features_tensor, targets_tensor = torch.stack(features), torch.stack(targets)
    return features_tensor, targets_tensor, names



def get_input_tensor_from_image_path(img_path, pipeline, device = "mps"):
    base, _ = os.path.splitext(img_path)
    img_path = f"{base}.png"

    # Prepare image and keypoints for the model
    img, keypoints = Augment.prepare_for_model(img_path, pipeline, [])
    x, _ = to_xy(img, keypoints)
    # We somehow have to unsqueeze twice, could be worth investigating
    x_tensor = x.unsqueeze(0).unsqueeze(0).to(device)  # Move to the correct device (MPS or CPU)
    return x_tensor




def build_error_data(kpd_model, data, device = "mps", norm_min = None, norm_max = None, clamp_threshold = 0.004):
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
        images, keypoints, names = get_full_unshuffled_batch(data, kpd_model.pipeline, augment_images=False)
        images = images.to(device)
        keypoints = keypoints.to(device)
        outputs = kpd_model(images)
        errors = (keypoints - outputs)**2
        mse_per_vector = torch.mean(errors, dim=1, keepdim=True)

        print("Before clamp")
        # Plot histogram using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(mse_per_vector.cpu().numpy(), bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of MSE Components', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Adds grid lines for better readability
        plt.show()
        
        mse_per_vector = torch.clamp(mse_per_vector, max=clamp_threshold)
        mse_per_vector_np = mse_per_vector.cpu().numpy()

        print("After clipping")
        # Plot histogram using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(mse_per_vector_np, bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of MSE Components', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Adds grid lines for better readability
        plt.show()
        
        """
        if not(norm_min == 0 and norm_max == 1):
            norm_min, norm_max = torch.min(mse_per_vector), torch.max(mse_per_vector)
        mse_per_vector = (mse_per_vector - norm_min)/(norm_max - norm_min)
        """

        # Step 1: Normalize by mean and variance
        mean = mse_per_vector.mean()
        std = mse_per_vector.std()
        normalized_vector = (mse_per_vector - mean) / std

        # Step 2: Scale to range [0, 1]
        min_val = normalized_vector.min()
        max_val = normalized_vector.max()
        scaled_errors = (normalized_vector - min_val) / (max_val - min_val)

        scaled_errors_np = scaled_errors.cpu().numpy()
        print("Normalized")
        # Plot histogram using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(scaled_errors_np, bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of MSE Components', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Adds grid lines for better readability
        plt.show()
    

    error_data = [{"Image Name": names[i], "Error": scaled_errors[i].item()} for i in range(len(names))]
    
    return error_data, 0, 0




def get_conf_batch_fast(error_data, pipeline, batchsize = 25, device = "mps"):

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
        img, keypoints = Augment.prepare_for_model(img_path, pipeline)

        x, y = to_xy(img, keypoints)
        features.append(x.unsqueeze(0))
        errors.append(torch.tensor([error]))
        names.append(name)

    features_tensor, error_tensor = torch.stack(features).to(device), torch.stack(errors).to(device)

    return features_tensor, error_tensor, names


"""
We want to normalize the KPD Model error to make training more reliable. So the following function aims to estimates 
max and min values of the error.
"""
def get_error_normalization_conf(kpd_model, data, batchsize = 100, augment_images = False, device = "mps"):
    kpd_model.eval()
    with torch.no_grad():
        images, errors, names = get_batch(data, kpd_model, batchsize=batchsize, augment_images=augment_images, device=device)
    
    min_error, max_error = torch.min(errors), torch.max(errors)
    normalizer = lambda error: (error-min_error)/(max_error-min_error)
    print("Computed error-normalization function.")
    return normalizer



"""
Fast version of train_conf_model. Uses the pre-computed errors (list of dicts with entries "Image Name" and "Error".)
"""
def train_conf_model_fast(conf_model, kpd_pipeline ,train_error_data, test_error_data, batchsize, test_batchsize, epochs, initial_lr = 1e-5, lr_decay = 0.99, device = "mps", augment_training_images = False, feedback_rate = 20, normalize_errors = True):
    best_test_loss = 10

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(conf_model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_decay)

    conf_model.train()

    # Note: might not be great conceptually to use the test data to find the normalization.

    for epoch in range(epochs):
        images, errors, names = get_conf_batch_fast(train_error_data, kpd_pipeline, batchsize=batchsize, augment_images=augment_training_images, device = device) 
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
                images, errors, names = get_conf_batch_fast(test_error_data, kpd_pipeline, batchsize = test_batchsize, augment_images = False, device = device)  
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



def get_input_tensor_from_image_path(model, img_path, device = "mps"):
    base, _ = os.path.splitext(img_path)
    img_path = f"{base}.png"

    # Prepare image and keypoints for the model
    img, keypoints = Augment.prepare_for_model(img_path, model.pipeline, [])
    x, _ = to_xy(img, keypoints)
    # We somehow have to unsqueeze twice, could be worth investigating
    x_tensor = x.unsqueeze(0).unsqueeze(0).to(device)  # Move to the correct device (MPS or CPU)
    return x_tensor



def get_original_image_pred(kpd_model, img_path, conf_model = None):
    input_tensor = get_input_tensor_from_image_path(kpd_model, img_path)
    pred = kpd_model(input_tensor)
    if conf_model is None:
        error_pred = None
    else:
        error_pred = conf_model(input_tensor).item()
    y1pred, x1pred, y2pred, x2pred = pred[0][0].item() * kpd_model.input_shape[0], pred[0][1].item() * kpd_model.input_shape[1], pred[0][2].item() * kpd_model.input_shape[0], pred[0][3].item() * kpd_model.input_shape[1]
    base, _ = os.path.splitext(img_path)
    img_path = f"{base}.png"
    original_img = Augment.quad_channel_2_single_channel(img_path)
    original_shape = original_img.shape
    
    pt1, pt2 = (x1pred, y1pred), (x2pred, y2pred)
    points_before_transform = Augment.reverse_infer_keypoints([pt1, pt2], original_shape, kpd_model.input_shape)
    return points_before_transform[0][0], points_before_transform[0][1], points_before_transform[1][0], points_before_transform[1][1], error_pred


def plot_model_prediction_on_original_image(kpd_model, img_path, conf_model = None):
    
    base, _ = os.path.splitext(img_path)
    img_path = f"{base}.png"
    original_img = Augment.quad_channel_2_single_channel(img_path)
    
    x1, y1, x2, y2, error_pred = get_original_image_pred(kpd_model, img_path, conf_model)
    p1, p2 = (x1, y1), (x2, y2)

    with_keypoints = ip.add_point_channels(original_img, p1, p2, radius=10)

    print("Normalized error prediction = ", error_pred)
    plt.plot()
    plt.imshow(with_keypoints)
    plt.show()
    print("------------------------")


