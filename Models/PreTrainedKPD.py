import torch.nn as nn
import torchvision.models as models
import Augment



class KeypointDetectionModel(nn.Module):
    def __init__(self, input_image_shape = (126, 256), input_channels=1, output_features=4):
        super(KeypointDetectionModel, self).__init__()

        self.input_shape = input_image_shape
        self.pipeline, self.augment_pipeline = Augment.create_transform_pipeline(False, self.input_shape), Augment.create_transform_pipeline(True, self.input_shape)


        # Load a pre-trained ResNet18 model
        self.base_model = models.resnet18(pretrained=True)

        # Modify the input layer to accept the specified number of input channels
        self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        """
        # Replace the fully connected layer to output the specified number of features
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, output_features)
        """

        # Replace the fully connected layer with a sequential block
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(256, output_features)
        )

    def forward(self, x):
        return self.base_model(x)

