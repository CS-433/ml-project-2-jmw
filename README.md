***Weber Length automatic measurement from ant images with integrated scale***


This software is divided in 3 main components:

- Create a dataset by manually providing keypoints on ant images with the script `Annotate.py`
- Keypoint Detection (KPD) and Confidence Models: from an ant image, estimate the coordinates of the two points that define Weber's length using Convolutional Neural Networks and provide an estimation of the error of the KPD model using the Confidence Model. We defined these models ourselves but got slighlty better results using modified larger pretrained models (PyTorch ResNet18 (18 layers)).
- With keypoints detected, what is left is to transform these two coordinates into a distance in milimeters using the integrated scale on the images. (Massi si tu peux donner plus de détails) 
- An interactive prediction script (More on that later)

You can reproduce our best model performance (for the pretrained resnet) with weights saved in `Models/Saves`. Non-standard used libraries and their versions are listed in `requirements.txt`.

To run the code, we recommend you first start by modifying the `Config.py` file, where you can specify the path of your training images, and so on to fit your setup. Also get confortable with conveniance methods of `Tools.py` such as `plot_model_prediction_on_original_image` with the `create_csv_from_predictions.ipynb` notebook.