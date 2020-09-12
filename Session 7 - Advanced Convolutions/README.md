## Analysis

### Target:
- change the code such that it uses GPU
- change the architecture to C1C2C3C40 (basically 3 MPs)
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 


### Result:
- Best Train accuracy: 88.15
- Best Test accuracy: 87.46
- Total Receptive Field: 98
- Total Parameters: 753,984
- Total Epochs 50

### Observation
- Model is good
- Acheive targeted accuracy within 10 epochs
- Model is not overfitting
- GhostBatchNormalization with L2 regularization provides quite good result.


## Losses and Acurracy plot:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%207%20-%20Advanced%20Convolutions/images/cifar_10_plot.png)

## Receptive field calculation
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%207%20-%20Advanced%20Convolutions/images/Cifar10_RF.PNG)
