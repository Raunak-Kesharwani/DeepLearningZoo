# Learning from this project 

For training the model on GPU we need to move our model to GPU using .to(device)  

Never call Optimizer before model otherwise it will not update and got stucked on forward pass  

Don't forgot shuffle = True in Dataloader , it may increase increase in model accuracy  

we can not pass value directly in nn.class because nn contains whatever they are class / constructor  

if using tanh activation than apply normalization  

totensor() not normalizes it standardises  

Normalization → prevents tanh saturation  

Sanity check shapes → print tensor sizes to catch silent bugs  
"""def forward(self, x):  
    print("Input:", x.shape)  
    x = self.layer_1(x)  
    print("After layer_1:", x.shape)  
    x = self.layer_2(x)  
    print("After layer_2:", x.shape)  
    x = self.layer_3(x)  
    print("After layer_3:", x.shape)  
    x = x.view(x.size(0), -1)  
    print("After flatten:", x.shape)  
    x = torch.tanh(self.fc_1(x))  
    x = self.fc_2(x)  
    return x"""  

torch.manual_seed(42) → fixes randomness on CPU  

torch.cuda.manual_seed_all(42) → fixes randomness on GPU  

cudnn.deterministic = True → forces same GPU operations each run  

cudnn.benchmark = False → prevents random algorithm selection  

# Observations 

output before solving saturation problem of tanh():  
    Accuracy: 96.4%  
    Avg loss: 0.119992  

output  after solving saturation problem of tanh():  
     Accuracy: 97.3%  
     Avg loss: 0.092725  

after adding momemtum = 0.9 in sgd :  
    # heavy jump in accuracy in first epoche to the 96%
    Accuracy: 98.6%  
    Avg loss: 0.042908  

changing activation from tanh to relu :  
    # heavy jump in accuracy in first epoche to the 96.5%  
    Accuracy: 98.9%
    Avg loss: 0.036310

changing optimizer from SGD to ADAM :  
    # for adma 0.01 learning rate is too high they expect 1-e3 or 1-e4  
    # heavy jump in accuracy in first epoche to the 97.1%  
    # don't know why training speed drops although training accuracy increased  
    # Training is unstablized  
    Accuracy: 99.0%  
    Avg loss: 0.033818  

##### Don't train anything i trained and saved the model just use it at inference 