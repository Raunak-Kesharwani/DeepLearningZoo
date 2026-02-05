#### ResNet18 (CIFAR-10, 224×224)

##### Overview
This model is a **ResNet-18–based image classifier** fine-tuned on the **CIFAR-10 dataset**, adapted to work with **224×224 RGB images**.  
It demonstrates how a deeper residual network behaves when trained on a small, low-resolution dataset but evaluated at a higher input resolution.

The model predicts one of **10 object categories** from a single input image.

---

##### Model Architecture
- **Backbone:** ResNet-18
- **Key idea:** Residual (skip) connections to ease training of deep networks
- **Classifier head:**
  - Dropout (p = 0.5)
  - Fully connected layer → 10 classes

Residual connections allow gradients to flow more easily during training, reducing vanishing-gradient issues compared to plain CNNs.

---

## Input Specification
- **Input shape:** `(3, 224, 224)`
- **Color space:** RGB
- **Preprocessing:**
  - Resize image to `224 × 224`
  - Normalize using:
    - Mean: `(0.485, 0.456, 0.406)`
    - Std: `(0.229, 0.224, 0.225)`

User-uploaded images can be of any size; resizing and normalization are handled before inference.

---

##### Output
- **Task:** Image classification
- **Number of classes:** 10
- **Classes:**
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

The model outputs:
- A **single predicted label**
- A **probability distribution** over all classes

---

##### Dataset
- **Dataset used for training:** CIFAR-10
- **Image content:** Natural images across 10 object categories
- **Original resolution:** 32×32 (upsampled during preprocessing for this model)

---

##### Inference Notes
- The model is exported using **TorchScript**
- Inference runs on **CPU**
- Model loading is **lazy** (loaded only when selected in the UI)

---

##### Why This Model Exists in the Zoo
This model is included to:
- Demonstrate **fine-tuning a standard backbone** (ResNet-18)
- Highlight the impact of **residual connections**
- Explore behavior when **training on small datasets** but **inferring at higher resolution**
- Serve as a reference for adapting pretrained architectures to custom datasets

---

##### Limitations
- CIFAR-10 images are low resolution by nature; upscaling to 224×224 does not add new information
- Performance is constrained by dataset size and diversity
- Not intended for real-world deployment; purely educational

---

##### Summary
This ResNet-18 CIFAR-10 model showcases how modern CNN architectures can be adapted, fine-tuned, and deployed in a clean inference pipeline using TorchScript and Streamlit.
