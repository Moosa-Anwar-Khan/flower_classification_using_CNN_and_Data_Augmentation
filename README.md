# Flower Classification using CNN and Data Augmentation

This repository contains the implementation of a flower classification task using Convolutional Neural Networks (CNN) and data augmentation techniques. The goal of this project was to classify images of flowers and improve the model's performance by applying data augmentation to enhance the training dataset.

## Problem Statement

The task involves classifying images of different flower species using deep learning techniques. The model is initially trained on a set of flower images, and the performance is evaluated based on the accuracy on the test dataset.

## Dataset

The dataset used for this task consists of flower images from the following source:

https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

## Approach

1. **Dataset**: The dataset consists of images of flowers, which are labeled according to the species of the flowers. The images are pre-processed and resized to ensure consistency in input size.

2. **Convolutional Neural Network (CNN)**: Initially, a CNN model was trained using the raw dataset. The model architecture consists of convolutional layers, pooling layers, and fully connected layers to classify the images into different flower species.

3. **Data Augmentation**: To improve the model's performance and reduce overfitting, data augmentation techniques were applied. These techniques include:
   - RandomRotation
   - RandomZoom
   - RandomFlip (Horizontal)

   These transformations artificially increase the size of the training dataset and help the model generalize better by providing diverse variations of the images.

4. **Model Evaluation**: The performance of the model was evaluated using the test dataset. The initial model's accuracy was 66%, and after applying data augmentation and dropout regularization, the accuracy improved to 73%.

## Results

- **Initial Model Accuracy (without Data Augmentation)**: 66%
- **Model Accuracy (with Data Augmentation)**: 73%

By using data augmentation, the model was able to learn more robust features, leading to improved performance on the testing dataset.

## Future Improvements

 - Apply more advanced augmentation techniques such as random cropping and color jittering.
 - Experiment with transfer learning to further improve accuracy.
 - Hyperparameter Tuning: The model can be further improved by tuning hyperparameters like learning rate, batch size, number of epochs, and the architecture itself (e.g., number of layers, filters, etc.). Tuning these parameters can lead to even better performance.


Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Pillow) 
- OpenCV