## Analysis

### Target:
- Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that

- Write a code which
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
- Best Train accuracy: 81.34
- Best Test accuracy: 83.43
- Total Parameters: 6,746,954
- Total Epochs 29

### Observation
- Model is good no overfitting
- We can acheive more good accuracy by fine tune some parameters


## Cyclic Learning rate of first 5 epochs
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2011%20-%20Super%20Convergence/visualization/lr_plot.png)

## Losses and Acurracy plot:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2011%20-%20Super%20Convergence/visualization/S11_plot.png)

## Correct classified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2011%20-%20Super%20Convergence/visualization/correct_classified_imgs.png)

## Misclassified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2011%20-%20Super%20Convergence/visualization/misclassified_imgs.png)

## 25 Misclassified images Grad-Cam:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%2011%20-%20Super%20Convergence/visualization/gradcam_misclassified_images.png)

## Train logs of 5 epochs with one cycle policy
: EPOCH= 0 Loss= 2.3157 Batch_id= 97 Accuracy= 11.76: 100%|██████████| 98/98 [00:24<00:00,  3.94it/s]

Test set: Average Loss= 0.0721 Batch_id= 312 Accuracy= 1218/10000 (12.18%)

: EPOCH= 1 Loss= 2.1709 Batch_id= 97 Accuracy= 15.55: 100%|██████████| 98/98 [00:25<00:00,  3.88it/s]

Test set: Average Loss= 0.0667 Batch_id= 312 Accuracy= 2072/10000 (20.72%)

: EPOCH= 2 Loss= 1.9219 Batch_id= 97 Accuracy= 23.75: 100%|██████████| 98/98 [00:24<00:00,  3.93it/s]

Test set: Average Loss= 0.0633 Batch_id= 312 Accuracy= 2345/10000 (23.45%)

: EPOCH= 3 Loss= 1.8447 Batch_id= 97 Accuracy= 31.55: 100%|██████████| 98/98 [00:25<00:00,  3.92it/s]

Test set: Average Loss= 0.0583 Batch_id= 312 Accuracy= 3162/10000 (31.62%)

: EPOCH= 4 Loss= 1.5637 Batch_id= 97 Accuracy= 37.72: 100%|██████████| 98/98 [00:25<00:00,  3.92it/s]

Test set: Average Loss= 0.0537 Batch_id= 312 Accuracy= 3623/10000 (36.23%)


## Training logs for remaining epochs:

: EPOCH= 5 Loss= 1.4663 Batch_id= 97 Accuracy= 43.70: 100%|██████████| 98/98 [00:24<00:00,  3.96it/s]

Test set: Average Loss= 0.0438 Batch_id= 312 Accuracy= 4920/10000 (49.20%)

: EPOCH= 6 Loss= 1.4625 Batch_id= 97 Accuracy= 46.51: 100%|██████████| 98/98 [00:24<00:00,  3.95it/s]

Test set: Average Loss= 0.0419 Batch_id= 312 Accuracy= 5095/10000 (50.95%)

: EPOCH= 7 Loss= 1.3343 Batch_id= 97 Accuracy= 48.78: 100%|██████████| 98/98 [00:24<00:00,  3.92it/s]

Test set: Average Loss= 0.0401 Batch_id= 312 Accuracy= 5307/10000 (53.07%)

: EPOCH= 8 Loss= 1.2984 Batch_id= 97 Accuracy= 51.07: 100%|██████████| 98/98 [00:24<00:00,  3.92it/s]

Test set: Average Loss= 0.0385 Batch_id= 312 Accuracy= 5489/10000 (54.89%)

: EPOCH= 9 Loss= 1.2006 Batch_id= 97 Accuracy= 53.21: 100%|██████████| 98/98 [00:24<00:00,  3.95it/s]

Test set: Average Loss= 0.0376 Batch_id= 312 Accuracy= 5621/10000 (56.21%)

: EPOCH= 10 Loss= 1.2122 Batch_id= 97 Accuracy= 56.03: 100%|██████████| 98/98 [00:25<00:00,  3.92it/s]

Test set: Average Loss= 0.0354 Batch_id= 312 Accuracy= 5847/10000 (58.47%)

: EPOCH= 11 Loss= 1.0992 Batch_id= 97 Accuracy= 58.85: 100%|██████████| 98/98 [00:25<00:00,  3.90it/s]

Test set: Average Loss= 0.0326 Batch_id= 312 Accuracy= 6284/10000 (62.84%)

: EPOCH= 12 Loss= 0.9695 Batch_id= 97 Accuracy= 60.94: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average Loss= 0.0298 Batch_id= 312 Accuracy= 6600/10000 (66.00%)

: EPOCH= 13 Loss= 0.9797 Batch_id= 97 Accuracy= 63.34: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]

Test set: Average Loss= 0.0299 Batch_id= 312 Accuracy= 6594/10000 (65.94%)

: EPOCH= 14 Loss= 0.9152 Batch_id= 97 Accuracy= 65.01: 100%|██████████| 98/98 [00:25<00:00,  3.87it/s]

Test set: Average Loss= 0.0282 Batch_id= 312 Accuracy= 6760/10000 (67.60%)

: EPOCH= 15 Loss= 0.9199 Batch_id= 97 Accuracy= 67.25: 100%|██████████| 98/98 [00:25<00:00,  3.79it/s]

Test set: Average Loss= 0.0272 Batch_id= 312 Accuracy= 6926/10000 (69.26%)

: EPOCH= 16 Loss= 0.9292 Batch_id= 97 Accuracy= 69.00: 100%|██████████| 98/98 [00:25<00:00,  3.78it/s]

Test set: Average Loss= 0.0245 Batch_id= 312 Accuracy= 7242/10000 (72.42%)

: EPOCH= 17 Loss= 0.7259 Batch_id= 97 Accuracy= 70.06: 100%|██████████| 98/98 [00:25<00:00,  3.80it/s]

Test set: Average Loss= 0.0233 Batch_id= 312 Accuracy= 7406/10000 (74.06%)

: EPOCH= 18 Loss= 0.7977 Batch_id= 97 Accuracy= 71.31: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

Test set: Average Loss= 0.0224 Batch_id= 312 Accuracy= 7460/10000 (74.60%)

: EPOCH= 19 Loss= 0.7428 Batch_id= 97 Accuracy= 72.82: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

Test set: Average Loss= 0.0233 Batch_id= 312 Accuracy= 7438/10000 (74.38%)

: EPOCH= 20 Loss= 0.7669 Batch_id= 97 Accuracy= 74.09: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]

Test set: Average Loss= 0.0195 Batch_id= 312 Accuracy= 7790/10000 (77.90%)

: EPOCH= 21 Loss= 0.6826 Batch_id= 97 Accuracy= 75.36: 100%|██████████| 98/98 [00:25<00:00,  3.78it/s]

Test set: Average Loss= 0.0198 Batch_id= 312 Accuracy= 7802/10000 (78.02%)

: EPOCH= 22 Loss= 0.7196 Batch_id= 97 Accuracy= 76.22: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

Test set: Average Loss= 0.0181 Batch_id= 312 Accuracy= 7928/10000 (79.28%)

: EPOCH= 23 Loss= 0.5787 Batch_id= 97 Accuracy= 77.62: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

Test set: Average Loss= 0.0170 Batch_id= 312 Accuracy= 8092/10000 (80.92%)

: EPOCH= 24 Loss= 0.6652 Batch_id= 97 Accuracy= 78.71: 100%|██████████| 98/98 [00:25<00:00,  3.77it/s]

Test set: Average Loss= 0.0167 Batch_id= 312 Accuracy= 8156/10000 (81.56%)

: EPOCH= 25 Loss= 0.5542 Batch_id= 97 Accuracy= 79.16: 100%|██████████| 98/98 [00:26<00:00,  3.75it/s]

Test set: Average Loss= 0.0183 Batch_id= 312 Accuracy= 7964/10000 (79.64%)

: EPOCH= 26 Loss= 0.6539 Batch_id= 97 Accuracy= 79.90: 100%|██████████| 98/98 [00:26<00:00,  3.76it/s]

Test set: Average Loss= 0.0151 Batch_id= 312 Accuracy= 8343/10000 (83.43%)

: EPOCH= 27 Loss= 0.5354 Batch_id= 97 Accuracy= 81.04: 100%|██████████| 98/98 [00:26<00:00,  3.76it/s]

Test set: Average Loss= 0.0152 Batch_id= 312 Accuracy= 8300/10000 (83.00%)

: EPOCH= 28 Loss= 0.6511 Batch_id= 97 Accuracy= 81.34: 100%|██████████| 98/98 [00:26<00:00,  3.72it/s]

Test set: Average Loss= 0.0161 Batch_id= 312 Accuracy= 8208/10000 (82.08%)
