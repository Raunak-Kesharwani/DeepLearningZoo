#### Learning Notes – ResNet18 (CIFAR-10, 224×224)

**num_workers parameter in DataLoader specifies the number of parallel CPU subprocesses used to preload data, preventing the GPU from waiting on data fetching and preprocessing.**  


##### Training Experience
I trained this model **without early stopping**, which meant training continued even after improvements saturated.  
Eventually, I **stopped training manually using the keyboard**, which highlighted the importance of proper training control logic in real experiments.

---

##### Understanding Residual Connections
This model helped me understand **why residual connections are essential** in deep networks.

Instead of forcing each layer to learn a full transformation, ResNet learns **residuals** (`F(x) + x`).  
This allows gradients to flow more easily, reduces training instability, and enables deeper networks to train effectively.

Residual connections directly solve the pain of depth in traditional CNNs.

---

##### Post-Training Issue (Important Lesson)
After training, I initially tried to **wrap the ResNet model inside another class** during inference.  
This changed the internal parameter names and caused **state_dict loading errors**, even though the architecture looked identical.

This taught me a critical lesson:
> **The model structure at inference time must exactly match the training-time structure.**

Even small wrapper changes can completely break checkpoint loading.

---

##### Key Takeaway
- Early stopping is essential for controlled training
- Residual connections make deep networks trainable
- Training-time and inference-time architectures must match exactly
- TorchScript enforces good deployment discipline
