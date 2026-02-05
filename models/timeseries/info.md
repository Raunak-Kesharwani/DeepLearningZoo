
### LSTM Sine Wave Predictor

#### Overview
This model is a **time-series forecasting model** based on an **LSTM (Long Short-Term Memory) network**.  
It is trained to predict the **next value in a sine wave sequence** given a fixed window of past values.

The project focuses on understanding how recurrent neural networks model **temporal dependencies** in sequential data.

---

#### Model Architecture
- **Model type:** LSTM
- **Input size:** 1 (scalar value per timestep)
- **Hidden size:** 64
- **Number of LSTM layers:** 1
- **Output layer:** Fully connected layer → 1 value

The LSTM processes the input sequence step by step and uses the **last hidden state** to predict the next value in the sequence.

---

#### Input Specification
- **Input shape:** `(20, 1)`
- **Batch shape during inference:** `(1, 20, 1)`
- **Meaning:**  
  - 20 consecutive time steps
  - Each step contains a single numeric value

Example input:
```
[x₀, x₁, x₂, ..., x₁₉]
```

---

#### Output
- **Output shape:** `(1, 1)`
- **Meaning:**  
  - Predicted value at the **next time step** (`x₂₀`)

The output is a **continuous scalar**, not a class or probability.

---

#### Dataset
- **Data type:** Synthetic sine wave
- **Generation formula:**
```
sin(0.02 * x) + noise
```
- **Noise:** Small Gaussian noise added to improve generalization

This setup provides a simple but effective way to study sequence modeling.

---

#### Inference Behavior
- Uses the **last 20 values** to predict the next value
- Runs on **CPU**
- Exported using **TorchScript** for deployment
- Model is loaded **lazily** when selected in the UI

---

#### Why This Model Exists in the Zoo
This model is included to:
- Demonstrate **sequence modeling** with LSTMs
- Show how temporal dependencies differ from image-based learning
- Serve as a foundation for more advanced models (GRU, multi-step forecasting)
- Compare recurrent models with feedforward approaches

---

#### Limitations
- Designed for **single-step prediction**
- Trained on a simple synthetic signal
- Not intended for real-world forecasting tasks

---

#### Summary
This LSTM-based sine wave predictor illustrates the core ideas behind recurrent neural networks and time-series forecasting, making it a practical and educational example within the model zoo.

