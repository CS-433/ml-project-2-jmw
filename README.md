** Weber Length automatic measurement from ant images with integrated scale **


This software is divided in 3 main components:

- Create a dataset by manually providing keypoints on ant images with the script Annotate.py
- Keypoint Detection (KPD) and Confidence Models: from an ant image, estimate the coordinates of the two points that define Weber's length using Convolutional Neural Networks and provide an estimation of the error of the KPD model using the Confidence Model. We defined these models ourselves but got slighlty better results using modified larger pretrained models (PyTorch ResNet18 (18 layers)).
- With keypoints detected, what is left is to transform these two coordinates into a distance in milimeters using the integrated scale on the images. (Massi si tu peux donner plus de détails) 

Our best model weights (for the pretrained resnet) are saves in Models/Saves.

To run the code, we recommend you first start by modifying the Config.py file, where you can specify the path of your training images, and so on to fit your setup.


