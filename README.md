***Weber Length automatic measurement from ant images with integrated scale***


This software is divided in 3 main components:

- Create a dataset by manually providing keypoints on ant images with the script `Annotate.py`
- Keypoint Detection (KPD) and Confidence Models: from an ant image, estimate the coordinates of the two points that define Weber's length using Convolutional Neural Networks and provide an estimation of the error of the KPD model using the Confidence Model. We defined these models ourselves but got slighlty better results using modified larger pretrained models (PyTorch ResNet18 (18 layers)).
- With keypoints detected, what is left is to transform these two coordinates into a distance in milimeters using the integrated scale on the images. We calculate a pixel-to-millimeter ratio using the scale bar visible in the image to ensure accurate real-world measurements. The script processes an input CSV file containing image paths and keypoint coordinates. Results are written to an output CSV by adding the real distance.

You can reproduce our best model performance (for the pretrained resnet) with weights saved in `Models/Saves` (pretrainedKPD_weights00166.pth, pretrainedConf_weights00166.pth). Non-standard used libraries and their versions are listed in `requirements.txt`.

To run the code, we recommend you first start by modifying the `Config.py` file, where you can specify the path of your training images, and so on to fit your setup. Also get confortable with conveniance methods of `Tools.py` such as `plot_model_prediction_on_original_image` with the `create_csv_from_predictions.ipynb` notebook.

**CSV Files**
All the CSV files are located in the CSVs folder.
- Combined.csv: Contains the image names along with their keypoints (used for training).
- combined_distance.csv: Contains the real distance of the images from the Combined.csv file.
- KPD_confidence_predictions.csv: Contains the image names along with their predicted keypoints (produced by the model).
- KPD_confidence_predictions_distance.csv: Contains the real distance of the images from the KPD_confidence_predictions.csv file.