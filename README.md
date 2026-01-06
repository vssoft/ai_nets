# Breast Cancer Detection using Ultrasound Images (BUSI)

This project utilizes deep learning techniques to detect breast cancer in women by analyzing ultrasound images. The goal is to classify the images into two categories: benign (non-cancerous) and malignant (cancerous).

## Dataset

The dataset used for training and testing the deep learning model is sourced from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data). It contains a total of 3 folders containing images from which 891 are 'benign', 421 are 'malignant' and 266 are 'normal'. The images are two types: containing original photos and contaiung the photo's manually made ground-truth masks. The images are in PNG format and are of various sizes.

## Usage

To use this project, clone the repository and install the required dependencies. The `train_unet_busi.py` file is used to train the model, and the `inspect_busi_predictions_gui.py` file is used to make predictions on new images. See the individual file comments for more information on usage.

You can customize training parameters such as learning rate, batch size, and number of epochs in the train_unet_busi.py script.

You can visualize predictions after successful training.

Model Architecture
The U-Net model consists of:

- Encoder: Downsampling path with convolutional layers and max pooling.
- Bottleneck: Deepest part of the network with the highest number of feature channels.
- Decoder: Upsampling path with transposed convolutions and skip connections.
- Output Layer: A 1x1 convolution to produce the final segmentation mask.

## Results

The deep learning model achieved high accuracy on the test set demonstrating the effectiveness of using deep learning for breast cancer detection.

Training Metrics:
- Loss: Combination of BCE and Dice Loss.
- IoU (Intersection over Union): Evaluated on the validation set.

Training takes about 10 min on a box using NVIDIA GeForce RTX 5070 Laptop GPU with 8 GB.

## Conclusion

This project demonstrates the potential of using deep learning for automated breast cancer detection using ultrasound images. The high accuracy achieved indicates that this approach could be a valuable tool for assisting radiologists in breast cancer diagnosis.

## Acknowledgements

- [Tilestats](https://www.tilestats.com/python-code/) for providing the original python code.
- [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data) for providing the dataset.


## Features

- **Custom U-Net Implementation**: Includes skip connections, encoder-decoder structure, and bottleneck layers.
- **Mixed Precision Training**: Supports Automatic Mixed Precision (AMP) for faster training and reduced VRAM usage.
- **Loss Functions**: Combines BCE and Dice Loss for better segmentation performance.
- **Checkpoints**: Automatically saves the best and latest model during training.
- **Visualization**: Tools to visualize predictions and ground truth masks.
- **Reproducibility**: Includes seed-setting for consistent results.

---

## Requirements

To run this project, you need the following:

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm

 ## License

This project is licensed under the MIT License.
