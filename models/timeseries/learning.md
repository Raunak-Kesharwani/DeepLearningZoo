LSTM results in format of output , (h_n,c_n).
ouput carries the same thing as h_n but in (batch , sequnce , hidden_Size) format. 
for accessing last hidden size from output use outputp[:,-1 ,:].

by default it uses BPTT for TBPTT we need to do manual changes.
follow this for training pipeline zero_grad → forward → loss → backward → clip → step.

Without pin_memory: 
CPU → OS paging → GPU (slow).

With pin_memory:
CPU (locked) → DMA → GPU (fast).


When You SHOULD Use pin_memory  

| Scenario          | Use it?     |
| ----------------- | ----------- |
| Training on GPU   | ✅ YES       |
| Large dataset     | ✅ YES       |
| Big batch size    | ✅ YES       |
| CPU-only training | ❌ No effect |
| Tiny dataset      | Optional    |

BEST PRACTICE (ASYNC COPY)  
X = X.to(device, non_blocking=True)  
y = y.to(device, non_blocking=True)  


This allows: 
CPU loads next batch  
GPU trains current batch  
True overlap of compute & transfer  
non_blocking=True only works if pin_memory=True  