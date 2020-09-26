## Analysis

### Target:
- Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
- Please make sure that your test_transforms are simple and only using ToTensor and Normalize
- Implement GradCam function as a module. 
- Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
- Target Accuracy is 87%

### Result:
- Best Train accuracy: 97.97
- Best Test accuracy: 90.63
- Total Parameters: 11,173,962
- Total Epochs 50

### Observation
- Model is overfitting
- Acheive targeted accuracy in 8 epochs
- Dropout  and GhostBatchNormalization are not used in model ResNet18.
- Hypothesis by using dropout and GhostBatchNormalization we be able to solve overfitting issue



## Losses and Acurracy plot:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%209%20-%20Image_Augmentation_and_Class_Activation_Maps/visualization/cifar_10_plot_using_resnet18_v4.png)

## Correct classified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%209%20-%20Image_Augmentation_and_Class_Activation_Maps/visualization/correct_classified_images.png)

## Misclassified images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%209%20-%20Image_Augmentation_and_Class_Activation_Maps/visualization/misclassified_images.png)

## Correct classified images activation map:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%209%20-%20Image_Augmentation_and_Class_Activation_Maps/visualization/correct_activation_map.png)

## Misclassified images activation images:
![](https://github.com/RaviVaishnav20/TSAI-EVA5/blob/master/Session%209%20-%20Image_Augmentation_and_Class_Activation_Maps/visualization/misclassified_activation_map.png)

## Accuracy and Losses logs:

: EPOCH= 0 Loss= 1.7647 Batch_id= 1562 Accuracy= 28.92: 100%|██████████| 1563/1563 [01:05<00:00, 23.88it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0528 Batch_id= 312 Accuracy= 3793/10000 (37.93%)

: EPOCH= 1 Loss= 1.3779 Batch_id= 1562 Accuracy= 41.91: 100%|██████████| 1563/1563 [01:05<00:00, 23.92it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0402 Batch_id= 312 Accuracy= 5327/10000 (53.27%)

: EPOCH= 2 Loss= 1.3197 Batch_id= 1562 Accuracy= 53.57: 100%|██████████| 1563/1563 [01:05<00:00, 23.99it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0331 Batch_id= 312 Accuracy= 6395/10000 (63.95%)

: EPOCH= 3 Loss= 0.9571 Batch_id= 1562 Accuracy= 60.88: 100%|██████████| 1563/1563 [01:05<00:00, 23.93it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0288 Batch_id= 312 Accuracy= 6773/10000 (67.73%)

: EPOCH= 4 Loss= 0.9124 Batch_id= 1562 Accuracy= 66.07: 100%|██████████| 1563/1563 [01:05<00:00, 23.94it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0265 Batch_id= 312 Accuracy= 7073/10000 (70.73%)

: EPOCH= 5 Loss= 0.7391 Batch_id= 1562 Accuracy= 69.79: 100%|██████████| 1563/1563 [01:05<00:00, 24.01it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0217 Batch_id= 312 Accuracy= 7581/10000 (75.81%)

: EPOCH= 6 Loss= 0.6810 Batch_id= 1562 Accuracy= 72.39: 100%|██████████| 1563/1563 [01:05<00:00, 23.95it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0234 Batch_id= 312 Accuracy= 7479/10000 (74.79%)

: EPOCH= 7 Loss= 0.4785 Batch_id= 1562 Accuracy= 74.30: 100%|██████████| 1563/1563 [01:05<00:00, 24.01it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0218 Batch_id= 312 Accuracy= 7598/10000 (75.98%)

: EPOCH= 8 Loss= 0.9736 Batch_id= 1562 Accuracy= 75.71: 100%|██████████| 1563/1563 [01:05<00:00, 24.04it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0180 Batch_id= 312 Accuracy= 8091/10000 (80.91%)

: EPOCH= 9 Loss= 1.0301 Batch_id= 1562 Accuracy= 76.80: 100%|██████████| 1563/1563 [01:05<00:00, 23.78it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0198 Batch_id= 312 Accuracy= 7857/10000 (78.57%)

: EPOCH= 10 Loss= 0.6526 Batch_id= 1562 Accuracy= 77.87: 100%|██████████| 1563/1563 [01:05<00:00, 23.89it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0174 Batch_id= 312 Accuracy= 8097/10000 (80.97%)

: EPOCH= 11 Loss= 0.3518 Batch_id= 1562 Accuracy= 78.68: 100%|██████████| 1563/1563 [01:05<00:00, 23.87it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0210 Batch_id= 312 Accuracy= 7725/10000 (77.25%)

: EPOCH= 12 Loss= 0.7924 Batch_id= 1562 Accuracy= 79.42: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0188 Batch_id= 312 Accuracy= 8003/10000 (80.03%)

: EPOCH= 13 Loss= 0.4039 Batch_id= 1562 Accuracy= 79.90: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0169 Batch_id= 312 Accuracy= 8162/10000 (81.62%)

: EPOCH= 14 Loss= 0.5357 Batch_id= 1562 Accuracy= 80.17: 100%|██████████| 1563/1563 [01:05<00:00, 23.94it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0167 Batch_id= 312 Accuracy= 8180/10000 (81.80%)

: EPOCH= 15 Loss= 0.4189 Batch_id= 1562 Accuracy= 81.08: 100%|██████████| 1563/1563 [01:05<00:00, 23.87it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0155 Batch_id= 312 Accuracy= 8360/10000 (83.60%)

: EPOCH= 16 Loss= 0.4700 Batch_id= 1562 Accuracy= 81.54: 100%|██████████| 1563/1563 [01:05<00:00, 23.96it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0152 Batch_id= 312 Accuracy= 8310/10000 (83.10%)

: EPOCH= 17 Loss= 0.4660 Batch_id= 1562 Accuracy= 81.81: 100%|██████████| 1563/1563 [01:05<00:00, 23.86it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0142 Batch_id= 312 Accuracy= 8484/10000 (84.84%)

: EPOCH= 18 Loss= 0.4630 Batch_id= 1562 Accuracy= 82.06: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0170 Batch_id= 312 Accuracy= 8210/10000 (82.10%)

: EPOCH= 19 Loss= 0.6926 Batch_id= 1562 Accuracy= 82.47: 100%|██████████| 1563/1563 [01:05<00:00, 24.01it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0169 Batch_id= 312 Accuracy= 8220/10000 (82.20%)

: EPOCH= 20 Loss= 0.6291 Batch_id= 1562 Accuracy= 82.30: 100%|██████████| 1563/1563 [01:05<00:00, 23.87it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0169 Batch_id= 312 Accuracy= 8208/10000 (82.08%)

: EPOCH= 21 Loss= 0.4279 Batch_id= 1562 Accuracy= 88.97: 100%|██████████| 1563/1563 [01:05<00:00, 23.89it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0100 Batch_id= 312 Accuracy= 8928/10000 (89.28%)

: EPOCH= 22 Loss= 0.3373 Batch_id= 1562 Accuracy= 90.66: 100%|██████████| 1563/1563 [01:05<00:00, 23.95it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0096 Batch_id= 312 Accuracy= 8957/10000 (89.57%)

: EPOCH= 23 Loss= 0.6747 Batch_id= 1562 Accuracy= 91.74: 100%|██████████| 1563/1563 [01:05<00:00, 23.98it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0096 Batch_id= 312 Accuracy= 8981/10000 (89.81%)

: EPOCH= 24 Loss= 0.3338 Batch_id= 1562 Accuracy= 92.25: 100%|██████████| 1563/1563 [01:05<00:00, 23.95it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0095 Batch_id= 312 Accuracy= 8992/10000 (89.92%)

: EPOCH= 25 Loss= 0.2507 Batch_id= 1562 Accuracy= 92.89: 100%|██████████| 1563/1563 [01:05<00:00, 23.95it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0095 Batch_id= 312 Accuracy= 9024/10000 (90.24%)

: EPOCH= 26 Loss= 0.0874 Batch_id= 1562 Accuracy= 93.19: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0095 Batch_id= 312 Accuracy= 8994/10000 (89.94%)

: EPOCH= 27 Loss= 0.3170 Batch_id= 1562 Accuracy= 93.68: 100%|██████████| 1563/1563 [01:05<00:00, 23.90it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0096 Batch_id= 312 Accuracy= 9007/10000 (90.07%)

: EPOCH= 28 Loss= 0.5854 Batch_id= 1562 Accuracy= 94.11: 100%|██████████| 1563/1563 [01:05<00:00, 23.99it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0097 Batch_id= 312 Accuracy= 9027/10000 (90.27%)

: EPOCH= 29 Loss= 0.1491 Batch_id= 1562 Accuracy= 94.36: 100%|██████████| 1563/1563 [01:05<00:00, 23.86it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0100 Batch_id= 312 Accuracy= 8994/10000 (89.94%)

: EPOCH= 30 Loss= 0.1806 Batch_id= 1562 Accuracy= 94.76: 100%|██████████| 1563/1563 [01:05<00:00, 23.85it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0100 Batch_id= 312 Accuracy= 9021/10000 (90.21%)

: EPOCH= 31 Loss= 0.0428 Batch_id= 1562 Accuracy= 95.03: 100%|██████████| 1563/1563 [01:05<00:00, 23.95it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0098 Batch_id= 312 Accuracy= 8990/10000 (89.90%)

: EPOCH= 32 Loss= 0.1733 Batch_id= 1562 Accuracy= 95.11: 100%|██████████| 1563/1563 [01:05<00:00, 23.93it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0105 Batch_id= 312 Accuracy= 8988/10000 (89.88%)

: EPOCH= 33 Loss= 0.1475 Batch_id= 1562 Accuracy= 95.38: 100%|██████████| 1563/1563 [01:05<00:00, 24.02it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0102 Batch_id= 312 Accuracy= 9005/10000 (90.05%)

: EPOCH= 34 Loss= 0.3801 Batch_id= 1562 Accuracy= 95.52: 100%|██████████| 1563/1563 [01:05<00:00, 23.84it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0104 Batch_id= 312 Accuracy= 8985/10000 (89.85%)

: EPOCH= 35 Loss= 0.2383 Batch_id= 1562 Accuracy= 95.62: 100%|██████████| 1563/1563 [01:05<00:00, 23.91it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0109 Batch_id= 312 Accuracy= 8960/10000 (89.60%)

: EPOCH= 36 Loss= 0.1350 Batch_id= 1562 Accuracy= 95.90: 100%|██████████| 1563/1563 [01:05<00:00, 23.87it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0105 Batch_id= 312 Accuracy= 8977/10000 (89.77%)

: EPOCH= 37 Loss= 0.0815 Batch_id= 1562 Accuracy= 95.90: 100%|██████████| 1563/1563 [01:05<00:00, 23.89it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0108 Batch_id= 312 Accuracy= 8994/10000 (89.94%)

: EPOCH= 38 Loss= 0.1705 Batch_id= 1562 Accuracy= 96.01: 100%|██████████| 1563/1563 [01:05<00:00, 23.96it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0103 Batch_id= 312 Accuracy= 9014/10000 (90.14%)

: EPOCH= 39 Loss= 0.0553 Batch_id= 1562 Accuracy= 96.26: 100%|██████████| 1563/1563 [01:04<00:00, 24.06it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0110 Batch_id= 312 Accuracy= 8943/10000 (89.43%)

: EPOCH= 40 Loss= 0.1224 Batch_id= 1562 Accuracy= 96.23: 100%|██████████| 1563/1563 [01:05<00:00, 23.93it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0106 Batch_id= 312 Accuracy= 8986/10000 (89.86%)

: EPOCH= 41 Loss= 0.0080 Batch_id= 1562 Accuracy= 96.12: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0113 Batch_id= 312 Accuracy= 8943/10000 (89.43%)

: EPOCH= 42 Loss= 0.0829 Batch_id= 1562 Accuracy= 96.86: 100%|██████████| 1563/1563 [01:05<00:00, 23.92it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0101 Batch_id= 312 Accuracy= 9047/10000 (90.47%)

: EPOCH= 43 Loss= 0.1192 Batch_id= 1562 Accuracy= 97.44: 100%|██████████| 1563/1563 [01:05<00:00, 23.90it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0100 Batch_id= 312 Accuracy= 9058/10000 (90.58%)

: EPOCH= 44 Loss= 0.0514 Batch_id= 1562 Accuracy= 97.58: 100%|██████████| 1563/1563 [01:05<00:00, 24.04it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0100 Batch_id= 312 Accuracy= 9049/10000 (90.49%)

: EPOCH= 45 Loss= 0.0130 Batch_id= 1562 Accuracy= 97.70: 100%|██████████| 1563/1563 [01:05<00:00, 23.97it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0102 Batch_id= 312 Accuracy= 9051/10000 (90.51%)

: EPOCH= 46 Loss= 0.0756 Batch_id= 1562 Accuracy= 97.74: 100%|██████████| 1563/1563 [01:05<00:00, 23.94it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0101 Batch_id= 312 Accuracy= 9063/10000 (90.63%)

: EPOCH= 47 Loss= 0.2118 Batch_id= 1562 Accuracy= 97.84: 100%|██████████| 1563/1563 [01:05<00:00, 23.93it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0102 Batch_id= 312 Accuracy= 9050/10000 (90.50%)

: EPOCH= 48 Loss= 0.0313 Batch_id= 1562 Accuracy= 97.95: 100%|██████████| 1563/1563 [01:05<00:00, 23.91it/s]
  0%|          | 0/1563 [00:00<?, ?it/s]
Test set: Average Loss= 0.0102 Batch_id= 312 Accuracy= 9055/10000 (90.55%)

: EPOCH= 49 Loss= 0.0436 Batch_id= 1562 Accuracy= 97.97: 100%|██████████| 1563/1563 [01:05<00:00, 24.05it/s]
Test set: Average Loss= 0.0103 Batch_id= 312 Accuracy= 9044/10000 (90.44%)
