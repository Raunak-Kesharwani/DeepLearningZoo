used perplexity loss : what is perplexity loss  - > perplexity = math.exp(avg_loss)

Perplexity loss :- On average, how many choices is the model confused between?
Examples:
| Loss | Perplexity | Interpretation             |
| ---- | ---------- | -------------------------- |
| 0.0  | 1.0        | Perfect prediction         |
| 0.69 | 2          | Choosing between 2 words   |
| 1.10 | 3          | Choosing between 3 words   |
| 2.30 | 10         | Choosing between 10 words  |
| 4.60 | 100        | Choosing between 100 words |

Look at loss curves for stability
Look at perplexity for interpretability


### GRU & LSTM Experiment Configuration

* **Dataset:** WikiText-2 (word-level language modeling)
* **Tokenizer:** WordLevel (Whitespace)
* **Sequence length:** 35
* **Batch size:** 64

**Model Architecture (Both):**

* Embedding dim: 256
* Hidden size: 256
* Layers: 2
* Dropout: 0.3 (between layers)
* Output: Linear (256 → vocab)

**Difference:**

* LSTM uses LSTM cells
* GRU uses GRU cells

**Training:**

* Loss: CrossEntropy
* Optimizer: AdamW (lr=3e-4, wd=1e-4)
* Gradient clipping: 1.0
* Epochs: 20
* Teacher forcing: 100%

**Metric:** Perplexity = exp(cross-entropy)

after each batch hidden state is initialized again 

while training both models with above parameters observed that both are started overfitting after 2nd epochs and results best model at same :  
for LSTM at second epoche perplexity is : 152 
for GRU at second epoche perplexity is : 155 



why observed overfitting :
Dataset is too small for model with above constraints (wikitext2 contains 2-million unique tokens only)
sequence length 35 is small and hidden size 256 with 2 layers makes them very capable to overfit 

now using this configuration :
hidden_size = 128
num_layers = 1
sequence_length = 60

got very poor results with both :
Perplexity score :
LSTM = 205
GRU = 195 
underfit because changed the sequence length from 35 to 60 and decreased layers by 1 

average time taken for each epoche for both GRU and LSTM is ~16 min 
LSTM is stopped at 13th epoche because of early stopping criteria has been hit 
GRU taken complete 20 epoche 

* Compared LSTM and GRU architectures for word-level language modeling on WikiText-2, observing that GRU achieved lower perplexity under identical training conditions due to faster convergence and parameter efficiency.

LSTM at 13 epochs perplexity score is 196.13
GRU at 13th epochs perplexity score is 181.73 
and converges to 20th epochs with perplexity score of 179 

GRU outperforms because data is limited and LSTM carries more gates so it takes more data to converge 
