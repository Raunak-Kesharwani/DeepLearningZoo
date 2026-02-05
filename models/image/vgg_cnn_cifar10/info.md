### VGG_CNN – CIFAR-10 Image Classification

#### 📌 What this model does

This model performs **image classification** on the **CIFAR-10 dataset**, which consists of small RGB images (32×32) belonging to 10 object categories such as animals and vehicles.

Given an input image, the model:
1. Resizes it to **32×32**
2. Processes it as an **RGB image (3 channels)**
3. Predicts which of the 10 classes the image belongs to
4. Outputs probabilities for all classes and the most likely label

---

#### 🧠 Model Architecture (High Level)

The model is inspired by **VGG-style CNNs**, characterized by:
- Stacked **convolutional layers**
- Small convolution kernels
- Gradual increase in feature depth
- Fully connected layers at the end for classification

Unlike LeNet (used for MNIST), this model:
- Operates on **color images**
- Learns more complex spatial and color-based features
- Requires deeper feature extraction due to dataset complexity

---

#### 🖼 Input Specification

- **Input shape:** `(3, 32, 32)`
- **Color mode:** RGB
- **Preprocessing:**
  - Resize to 32×32
  - Normalize using:
    - Mean: `(0.485, 0.456, 0.406)`
    - Std: `(0.229, 0.224, 0.225)`

User-uploaded images of any size or format are automatically adapted to this format before inference.

---

#### 🏷 Classes

The model predicts one of the following 10 classes:

- airplane  
- automobile  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

---

#### 📤 Output

- **Top-1 predicted class**
- **Probability distribution** across all 10 classes

This helps understand not only *what* the model predicts, but also *how confident* it is.

---

#### ⚙️ Inference Details

- Framework: **PyTorch**
- Inference format: **TorchScript**
- Device: **CPU**
- Model is loaded lazily when selected in the UI

---

#### 🎯 Purpose of this Model in the Zoo

This model demonstrates:
- Transition from grayscale to **RGB image modeling**
- Increased architectural depth compared to MNIST models
- How preprocessing and normalization become more important for real-world images
- A scalable pattern for adding more complex CNNs to the model zoo
