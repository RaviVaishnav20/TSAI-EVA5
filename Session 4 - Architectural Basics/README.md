## Target
 - 99.4% validation accuracy
 - Less than 20k Parameters
 - Less than 20 Epochs
 - No fully connected layer

## We can use below concept to achieve the target
How many layers,
MaxPooling,
1x1 Convolutions,
3x3 Convolutions,
Receptive Field,
SoftMax,
Learning Rate,
Kernels and how do we decide the number of kernels?
Batch Normalization,
Image Normalization,
Position of MaxPooling,
Concept of Transition Layers,
Position of Transition Layer,
DropOut
When do we introduce DropOut, or when do we know we have some overfitting
The distance of MaxPooling from Prediction,
The distance of Batch Normalization from Prediction,
When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
How do we know our network is not going well, comparatively, very early
Batch Size, and effects of batch size

## Model Summmary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 128, 28, 28]           1,280
       BatchNorm2d-2          [-1, 128, 28, 28]             256
         Dropout2d-3          [-1, 128, 28, 28]               0
            Conv2d-4            [-1, 8, 30, 30]           1,032
            Conv2d-5           [-1, 16, 30, 30]           1,168
       BatchNorm2d-6           [-1, 16, 30, 30]              32
         Dropout2d-7           [-1, 16, 30, 30]               0
         MaxPool2d-8           [-1, 16, 15, 15]               0
            Conv2d-9           [-1, 16, 15, 15]           2,320
      BatchNorm2d-10           [-1, 16, 15, 15]              32
        Dropout2d-11           [-1, 16, 15, 15]               0
           Conv2d-12           [-1, 32, 15, 15]           4,640
      BatchNorm2d-13           [-1, 32, 15, 15]              64
        Dropout2d-14           [-1, 32, 15, 15]               0
        MaxPool2d-15             [-1, 32, 7, 7]               0
           Conv2d-16             [-1, 16, 9, 9]             528
           Conv2d-17             [-1, 16, 9, 9]           2,320
      BatchNorm2d-18             [-1, 16, 9, 9]              32
        Dropout2d-19             [-1, 16, 9, 9]               0
           Conv2d-20             [-1, 32, 9, 9]           4,640
      BatchNorm2d-21             [-1, 32, 9, 9]              64
        Dropout2d-22             [-1, 32, 9, 9]               0
           Conv2d-23           [-1, 10, 11, 11]             330
        AvgPool2d-24             [-1, 10, 1, 1]               0
================================================================
Total params: 18,738
Trainable params: 18,738
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 3.08
Params size (MB): 0.07
Estimated Total Size (MB): 3.15
----------------------------------------------------------------

## Training Loss

  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:53: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
epoch: 1 loss=0.27045467495918274 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.58it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.1221, Accuracy: 9685/10000 (96.8%)

epoch: 2 loss=0.09988906979560852 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.15it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0604, Accuracy: 9823/10000 (98.2%)

epoch: 3 loss=0.20125557482242584 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0480, Accuracy: 9843/10000 (98.4%)

epoch: 4 loss=0.0712851956486702 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.22it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0371, Accuracy: 9890/10000 (98.9%)

epoch: 5 loss=0.04961127042770386 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.45it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0321, Accuracy: 9897/10000 (99.0%)

epoch: 6 loss=0.054023560136556625 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.16it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0271, Accuracy: 9913/10000 (99.1%)

epoch: 7 loss=0.07397448271512985 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.32it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0273, Accuracy: 9909/10000 (99.1%)

epoch: 8 loss=0.05811620131134987 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.65it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0239, Accuracy: 9928/10000 (99.3%)

epoch: 9 loss=0.08609984070062637 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.86it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0222, Accuracy: 9930/10000 (99.3%)

epoch: 10 loss=0.10347550362348557 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.04it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0234, Accuracy: 9921/10000 (99.2%)

epoch: 11 loss=0.10419472306966782 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.88it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9930/10000 (99.3%)

epoch: 12 loss=0.004044002387672663 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.97it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0223, Accuracy: 9930/10000 (99.3%)

epoch: 13 loss=0.05143119767308235 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.56it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0201, Accuracy: 9930/10000 (99.3%)

epoch: 14 loss=0.03383662924170494 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.86it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0187, Accuracy: 9940/10000 (99.4%)

epoch: 15 loss=0.037076253443956375 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.42it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0209, Accuracy: 9935/10000 (99.3%)

epoch: 16 loss=0.009786871261894703 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.50it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0190, Accuracy: 9944/10000 (99.4%)

epoch: 17 loss=0.024468591436743736 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.36it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0177, Accuracy: 9946/10000 (99.5%)

epoch: 18 loss=0.030203601345419884 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.40it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0171, Accuracy: 9937/10000 (99.4%)

epoch: 19 loss=0.04640066251158714 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.72it/s]

Test set: Average loss: 0.0179, Accuracy: 9938/10000 (99.4%)
