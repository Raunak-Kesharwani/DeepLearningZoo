# LeNet-5 on MNIST — Project Overview

This project implements the classic **LeNet-5** convolutional neural network for handwritten digit classification on the **MNIST dataset**.

The model takes a **32×32 grayscale image** as input and predicts one of **10 digit classes (0–9)**.

## Model Details

- **Architecture**: LeNet-5
- **Input shape**: (1, 32, 32)
- **Dataset**: MNIST
- **Task**: Image Classification
- **Number of classes**: 10
- **Output**: Class logits → Softmax probabilities

## Inference Notes

- The model is already **trained and saved**
- During deployment:
  - The raw PyTorch model is converted to **TorchScript**
  - Only **inference** is performed
  - No training or fine-tuning happens in the app
- User-uploaded images (any size / RGB or grayscale) are:
  - Converted to grayscale
  - Resized to 32×32
  - Normalized using MNIST statistics

This project serves as a **baseline image-classification model** and a reference for building, debugging, and deploying CNNs in PyTorch.
