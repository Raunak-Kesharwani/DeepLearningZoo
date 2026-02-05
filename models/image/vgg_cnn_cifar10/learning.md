#### Learning from this project 

Trained Deep CNN without normlization with adam in VGG style  

we can not pass value directly in nn.class because nn contains whatever they are class / constructor  

use nn.dropout() instead of nn.dropout1d() this is for CNN1d 
use dropout after activation (standard good practice)
weight decay standard for Sgd 5e-4

#### Observations 

Observed when training started test accuracy is started from 45% when using SGD :  
    Accuracy: 79.6%  
    Avg loss: 101.463741  
    # this is the clearly case of overfitting training accuracy is too high(95%) training_loss is low but better than adam with no normalization  
    # but test_accuracy is low and test_loss is high   

obsereved training unstabilty may be because of adam optimizer :
    Accuracy: 78.0%  
    Avg loss: 164.200599  
    # this is the clearly case of overfitting training accuracy is too high(95%) training_loss is low with no normalization
    # but test_accuracy is low and test_loss is high 

#### After normalization 
with SGD got :
    best validation Accuracy: 84.3%    
    lowest Avg loss: 101.387872    
    but training accuracy is still high means we can still impove model    
    # after adding L2 regularization   

with Adam got :  
    best validation Accuracy: 83%   
    but training accuracy is still high means we can still impove model    
    got with lr 2e-3  
     Accuracy: 84.1%  
    Avg loss: 104.685413  
    # after adding L2 regularization  
    Accuracy: 84.7%  
    Avg loss: 85.936681  

#### After adding correct dropout nn.Dropout()  

got to see imrovement   
with adam got :  
    best validation accuracy : 85.7%  

after applying on the fly augumenttion got to see imporve by 1% and overfitting is completely reduced   
    Accuracy: 86.0%  
    Avg loss: 64.976720  

after incresing epoches to 80 :  
    Accuracy: 91.5%   
    Avg loss: 51.88400  
 
#### key Learnings

Data augmentation is a more effective approach than dropout for vision models and focuses on learning invariance rather than memorization.  
Preventing co-adaptation is the goal of dropout; excessive use of it usually conceals architectural problems.  
SGD and Adam/AdamW optimizers have different weight decay values.  
Long training runs are ideal for using cosine learning rate schedules, which enhance overall convergence.  
Confidence overfitting is indicated by an increase in validation loss while maintaining accuracy, but this is not always a problem.  
The best model is not always the last one trained; it is important to stop early.  
Regularization maintains training stability, architecture sets the performance bar, and optimization helps you reach it.  
You cannot optimize speed and reproducibility at the same time when comparing the cuDNN benchmark to deterministic.  

