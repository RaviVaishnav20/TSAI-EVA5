## Analysis

### Target:
- Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that

-Write a code which
  - uses this new ResNet Architecture for Cifar10:
   ``` PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
Target Accuracy: 90%. 
The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you'd get. 
```
 

### Result:
- Best Train accuracy: 98.40
- Best Test accuracy: 91.77
- Total Parameters: 11,173,962
- Total Epochs 50

### Observation
- Model is overfitting
- Acheive targeted accuracy in 25 epochs
- LR finder helps to find better learning rate 
- Cutout help to learn model better



## Losses and Acurracy plot:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2010%20-%20Advanced%20Concepts%20in%20Training%20%26%20Learning%20Rates/visualization/S10_plot.png)

## Correct classified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2010%20-%20Advanced%20Concepts%20in%20Training%20%26%20Learning%20Rates/visualization/correct_classified_imgs%20(1).png)

## Misclassified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2010%20-%20Advanced%20Concepts%20in%20Training%20%26%20Learning%20Rates/visualization/misclassified_imgs%20(1).png)

## 25 Misclassified images Grad-Cam:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2010%20-%20Advanced%20Concepts%20in%20Training%20%26%20Learning%20Rates/visualization/gradcam_misclassified_images.png)

## Training logs:

: EPOCH= 0 Loss= 1.5031 Batch_id= 781 Accuracy= 46.04: 100%|██████████| 782/782 [01:02<00:00, 12.53it/s]

Test set: Average Loss= 0.0330 Batch_id= 312 Accuracy= 6137/10000 (61.37%)

: EPOCH= 1 Loss= 1.0756 Batch_id= 781 Accuracy= 64.42: 100%|██████████| 782/782 [01:02<00:00, 12.55it/s]

Test set: Average Loss= 0.0287 Batch_id= 312 Accuracy= 6957/10000 (69.57%)

: EPOCH= 2 Loss= 0.7778 Batch_id= 781 Accuracy= 71.76: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0220 Batch_id= 312 Accuracy= 7605/10000 (76.05%)

: EPOCH= 3 Loss= 0.4762 Batch_id= 781 Accuracy= 76.01: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0203 Batch_id= 312 Accuracy= 7859/10000 (78.59%)

: EPOCH= 4 Loss= 0.6324 Batch_id= 781 Accuracy= 78.54: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0169 Batch_id= 312 Accuracy= 8181/10000 (81.81%)

: EPOCH= 5 Loss= 0.6491 Batch_id= 781 Accuracy= 80.50: 100%|██████████| 782/782 [01:02<00:00, 12.55it/s]

Test set: Average Loss= 0.0170 Batch_id= 312 Accuracy= 8167/10000 (81.67%)

: EPOCH= 6 Loss= 0.5261 Batch_id= 781 Accuracy= 82.18: 100%|██████████| 782/782 [01:02<00:00, 12.52it/s]

Test set: Average Loss= 0.0153 Batch_id= 312 Accuracy= 8348/10000 (83.48%)

: EPOCH= 7 Loss= 0.5723 Batch_id= 781 Accuracy= 83.56: 100%|██████████| 782/782 [01:02<00:00, 12.60it/s]

Test set: Average Loss= 0.0158 Batch_id= 312 Accuracy= 8294/10000 (82.94%)

: EPOCH= 8 Loss= 0.2602 Batch_id= 781 Accuracy= 84.73: 100%|██████████| 782/782 [01:02<00:00, 12.61it/s]

Test set: Average Loss= 0.0139 Batch_id= 312 Accuracy= 8472/10000 (84.72%)

: EPOCH= 9 Loss= 0.5013 Batch_id= 781 Accuracy= 85.26: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0141 Batch_id= 312 Accuracy= 8483/10000 (84.83%)

: EPOCH= 10 Loss= 0.4382 Batch_id= 781 Accuracy= 86.41: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0142 Batch_id= 312 Accuracy= 8461/10000 (84.61%)

: EPOCH= 11 Loss= 0.2270 Batch_id= 781 Accuracy= 87.20: 100%|██████████| 782/782 [01:02<00:00, 12.55it/s]

Test set: Average Loss= 0.0156 Batch_id= 312 Accuracy= 8429/10000 (84.29%)

: EPOCH= 12 Loss= 0.4515 Batch_id= 781 Accuracy= 87.97: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0146 Batch_id= 312 Accuracy= 8491/10000 (84.91%)

: EPOCH= 13 Loss= 1.1740 Batch_id= 781 Accuracy= 88.33: 100%|██████████| 782/782 [01:02<00:00, 12.57it/s]

Test set: Average Loss= 0.0126 Batch_id= 312 Accuracy= 8637/10000 (86.37%)

: EPOCH= 14 Loss= 0.2645 Batch_id= 781 Accuracy= 89.14: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0139 Batch_id= 312 Accuracy= 8568/10000 (85.68%)

: EPOCH= 15 Loss= 0.0536 Batch_id= 781 Accuracy= 89.67: 100%|██████████| 782/782 [01:02<00:00, 12.57it/s]

Test set: Average Loss= 0.0139 Batch_id= 312 Accuracy= 8576/10000 (85.76%)

: EPOCH= 16 Loss= 0.1357 Batch_id= 781 Accuracy= 90.16: 100%|██████████| 782/782 [01:02<00:00, 12.53it/s]

Test set: Average Loss= 0.0147 Batch_id= 312 Accuracy= 8515/10000 (85.15%)

: EPOCH= 17 Loss= 0.5336 Batch_id= 781 Accuracy= 90.90: 100%|██████████| 782/782 [01:02<00:00, 12.59it/s]

Test set: Average Loss= 0.0162 Batch_id= 312 Accuracy= 8434/10000 (84.34%)

: EPOCH= 18 Loss= 0.3367 Batch_id= 781 Accuracy= 91.13: 100%|██████████| 782/782 [01:02<00:00, 12.56it/s]

Test set: Average Loss= 0.0120 Batch_id= 312 Accuracy= 8759/10000 (87.59%)

: EPOCH= 19 Loss= 0.2535 Batch_id= 781 Accuracy= 91.27: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0125 Batch_id= 312 Accuracy= 8706/10000 (87.06%)

: EPOCH= 20 Loss= 0.1557 Batch_id= 781 Accuracy= 91.66: 100%|██████████| 782/782 [01:01<00:00, 12.61it/s]

Test set: Average Loss= 0.0132 Batch_id= 312 Accuracy= 8730/10000 (87.30%)

: EPOCH= 21 Loss= 0.6419 Batch_id= 781 Accuracy= 92.25: 100%|██████████| 782/782 [01:01<00:00, 12.63it/s]

Test set: Average Loss= 0.0123 Batch_id= 312 Accuracy= 8770/10000 (87.70%)

: EPOCH= 22 Loss= 0.3342 Batch_id= 781 Accuracy= 92.22: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0136 Batch_id= 312 Accuracy= 8673/10000 (86.73%)

: EPOCH= 23 Loss= 0.3083 Batch_id= 781 Accuracy= 92.54: 100%|██████████| 782/782 [01:02<00:00, 12.59it/s]

Test set: Average Loss= 0.0131 Batch_id= 312 Accuracy= 8736/10000 (87.36%)

: EPOCH= 24 Loss= 0.1081 Batch_id= 781 Accuracy= 93.01: 100%|██████████| 782/782 [01:02<00:00, 12.60it/s]

Test set: Average Loss= 0.0118 Batch_id= 312 Accuracy= 8808/10000 (88.08%)

: EPOCH= 25 Loss= 0.1609 Batch_id= 781 Accuracy= 93.07: 100%|██████████| 782/782 [01:02<00:00, 12.56it/s]

Test set: Average Loss= 0.0119 Batch_id= 312 Accuracy= 8807/10000 (88.07%)

: EPOCH= 26 Loss= 0.1255 Batch_id= 781 Accuracy= 93.21: 100%|██████████| 782/782 [01:01<00:00, 12.62it/s]

Test set: Average Loss= 0.0124 Batch_id= 312 Accuracy= 8758/10000 (87.58%)

: EPOCH= 27 Loss= 0.1013 Batch_id= 781 Accuracy= 93.38: 100%|██████████| 782/782 [01:02<00:00, 12.56it/s]

Test set: Average Loss= 0.0125 Batch_id= 312 Accuracy= 8822/10000 (88.22%)

: EPOCH= 28 Loss= 0.1971 Batch_id= 781 Accuracy= 93.54: 100%|██████████| 782/782 [01:02<00:00, 12.55it/s]

Test set: Average Loss= 0.0124 Batch_id= 312 Accuracy= 8807/10000 (88.07%)

: EPOCH= 29 Loss= 0.1646 Batch_id= 781 Accuracy= 93.63: 100%|██████████| 782/782 [01:02<00:00, 12.59it/s]

Test set: Average Loss= 0.0122 Batch_id= 312 Accuracy= 8809/10000 (88.09%)

: EPOCH= 30 Loss= 0.4540 Batch_id= 781 Accuracy= 93.86: 100%|██████████| 782/782 [01:02<00:00, 12.60it/s]

Test set: Average Loss= 0.0132 Batch_id= 312 Accuracy= 8736/10000 (87.36%)

: EPOCH= 31 Loss= 0.1497 Batch_id= 781 Accuracy= 94.16: 100%|██████████| 782/782 [01:02<00:00, 12.57it/s]

Test set: Average Loss= 0.0123 Batch_id= 312 Accuracy= 8784/10000 (87.84%)

: EPOCH= 32 Loss= 0.1567 Batch_id= 781 Accuracy= 94.03: 100%|██████████| 782/782 [01:02<00:00, 12.61it/s]

Test set: Average Loss= 0.0124 Batch_id= 312 Accuracy= 8820/10000 (88.20%)

: EPOCH= 33 Loss= 0.2823 Batch_id= 781 Accuracy= 94.27: 100%|██████████| 782/782 [01:01<00:00, 12.62it/s]

Test set: Average Loss= 0.0116 Batch_id= 312 Accuracy= 8887/10000 (88.87%)

: EPOCH= 34 Loss= 0.0244 Batch_id= 781 Accuracy= 94.27: 100%|██████████| 782/782 [01:02<00:00, 12.52it/s]

Test set: Average Loss= 0.0113 Batch_id= 312 Accuracy= 8883/10000 (88.83%)

: EPOCH= 35 Loss= 0.3660 Batch_id= 781 Accuracy= 94.26: 100%|██████████| 782/782 [01:02<00:00, 12.57it/s]

Test set: Average Loss= 0.0133 Batch_id= 312 Accuracy= 8741/10000 (87.41%)

: EPOCH= 36 Loss= 0.3643 Batch_id= 781 Accuracy= 94.29: 100%|██████████| 782/782 [01:02<00:00, 12.55it/s]

Test set: Average Loss= 0.0126 Batch_id= 312 Accuracy= 8832/10000 (88.32%)

: EPOCH= 37 Loss= 0.1814 Batch_id= 781 Accuracy= 94.35: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0123 Batch_id= 312 Accuracy= 8812/10000 (88.12%)

: EPOCH= 38 Loss= 0.0710 Batch_id= 781 Accuracy= 94.54: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0126 Batch_id= 312 Accuracy= 8802/10000 (88.02%)

: EPOCH= 39 Loss= 0.0499 Batch_id= 781 Accuracy= 94.44: 100%|██████████| 782/782 [01:02<00:00, 12.60it/s]

Test set: Average Loss= 0.0124 Batch_id= 312 Accuracy= 8799/10000 (87.99%)

: EPOCH= 40 Loss= 0.0561 Batch_id= 781 Accuracy= 94.55: 100%|██████████| 782/782 [01:02<00:00, 12.52it/s]

Test set: Average Loss= 0.0122 Batch_id= 312 Accuracy= 8821/10000 (88.21%)

: EPOCH= 41 Loss= 0.6596 Batch_id= 781 Accuracy= 94.68: 100%|██████████| 782/782 [01:02<00:00, 12.57it/s]

Test set: Average Loss= 0.0120 Batch_id= 312 Accuracy= 8846/10000 (88.46%)

: EPOCH= 42 Loss= 0.1878 Batch_id= 781 Accuracy= 94.68: 100%|██████████| 782/782 [01:02<00:00, 12.56it/s]

Test set: Average Loss= 0.0115 Batch_id= 312 Accuracy= 8907/10000 (89.07%)

: EPOCH= 43 Loss= 0.0526 Batch_id= 781 Accuracy= 94.78: 100%|██████████| 782/782 [01:02<00:00, 12.60it/s]

Test set: Average Loss= 0.0127 Batch_id= 312 Accuracy= 8800/10000 (88.00%)

: EPOCH= 44 Loss= 0.1352 Batch_id= 781 Accuracy= 94.82: 100%|██████████| 782/782 [01:02<00:00, 12.58it/s]

Test set: Average Loss= 0.0121 Batch_id= 312 Accuracy= 8852/10000 (88.52%)

: EPOCH= 45 Loss= 0.0518 Batch_id= 781 Accuracy= 94.69: 100%|██████████| 782/782 [01:02<00:00, 12.54it/s]

Test set: Average Loss= 0.0115 Batch_id= 312 Accuracy= 8840/10000 (88.40%)

: EPOCH= 46 Loss= 0.0889 Batch_id= 781 Accuracy= 96.95: 100%|██████████| 782/782 [01:02<00:00, 12.53it/s]

Test set: Average Loss= 0.0091 Batch_id= 312 Accuracy= 9108/10000 (91.08%)

: EPOCH= 47 Loss= 0.2039 Batch_id= 781 Accuracy= 97.93: 100%|██████████| 782/782 [01:02<00:00, 12.61it/s]

Test set: Average Loss= 0.0089 Batch_id= 312 Accuracy= 9125/10000 (91.25%)

: EPOCH= 48 Loss= 0.0468 Batch_id= 781 Accuracy= 98.29: 100%|██████████| 782/782 [01:02<00:00, 12.56it/s]

Test set: Average Loss= 0.0089 Batch_id= 312 Accuracy= 9153/10000 (91.53%)

: EPOCH= 49 Loss= 0.0738 Batch_id= 781 Accuracy= 98.40: 100%|██████████| 782/782 [01:02<00:00, 12.53it/s]

Test set: Average Loss= 0.0088 Batch_id= 312 Accuracy= 9177/10000 (91.77%)

