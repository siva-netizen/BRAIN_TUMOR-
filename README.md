

# Brain Tumor Prediction

This project leverages deep learning to classify brain tumor images into four distinct classes using grayscale images. The trained model is integrated with a Flask web application to enable real-time predictions.  

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Features](#features)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)


---

## Overview
The **Brain Tumor Prediction** system is designed to classify MRI images into four categories, indicating different brain tumor types. The project employs Convolutional Neural Networks (CNNs) for training, validation, and testing. Early stopping is implemented to optimize training by avoiding overfitting.  

## Dataset
- The dataset contains MRI scans in grayscale format.
- The training, validation, and testing data are divided as follows:
  - Training set: 4,571 images
  - Validation set: 1,141 images
  - Test set: 1,311 images
- Images are resized to **256x256 pixels**.

## Model Architecture
The model is built using the Keras Sequential API. Below are the details of the architecture:
- **Input Layer**: Accepts grayscale images of size `(256, 256)`.
- **Convolution Layers**: Convolutional layers with filters for feature extraction.
- **Batch Normalization**: Applied for faster convergence.
- **Pooling Layers**: MaxPooling2D for dimensionality reduction.
- **Dropout Layers**: Regularization to prevent overfitting.
- **Fully Connected Layers**: Dense layers for classification.
- **Activation Function**: `relu` for hidden layers and `softmax` for the output layer.

### Optimization
- **Early Stopping**: Stops training when validation loss stops improving.
- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam optimizer.

## Results
The training and validation loss/accuracy curves show:
- Best epoch for validation loss: 21  
- Best epoch for validation accuracy: 26  

### Performance Metrics:
- Achieved high validation accuracy.
- Test set accuracy and confusion matrix will be generated on deployment.

## Features
- **Real-Time Predictions**: A Flask web application allows users to upload MRI images for predictions.
- **Dynamic Visualization**: Displays results with probabilities.
- **Model Integration**: Easily extendable to include more classes or augment data.

## Usage
1. Run the Flask app:
   ```bash
   flask run
   ```
2. Upload an MRI image in the web interface.
3. View predictions and probabilities on the results page.

## Technologies Used
- **Deep Learning Framework**: TensorFlow/Keras
- **Backend**: Flask
- **Visualization**: Matplotlib
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML, CSS (Flask Templates)

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.
