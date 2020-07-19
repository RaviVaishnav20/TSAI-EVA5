## Neural Architecture

### Importing Libraries 

`from __future__ import print_function`

First of all, from __future__ import print_function needs to be the first line of code in your script (aside from some exceptions mentioned below). Second of all, as other answers have said, you have to use print as a function now. That's the whole point of from __future__ import print_function; to bring the print function from Python 3 into Python 2.6+

`import torch`

Import torch to work with PyTorch and perform the operation

`import torch.nn as nn`

torch.nn provide us many more classes and modules to implement and train the neural network.

`import torch.nn.functional as F`

torch.nn.functional is give access to call any function inside torch.nn

**Note:**

nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
nn.functional.relu on the other side is just the functional API call to the relu function, so that you can add it e.g. in your forward method yourself.

Generally speaking it might depend on your coding style if you prefer modules for the activations or the functional calls

`import torch.optim as optim`

torch.optim is a package implementing various optimization algorithms. 

`from torchvision import datasets, transforms`

torchvision contains most of the datasets and architectures used in Neural Networks.
-torchvision.datasets : Data loaders for popular vision datasets
-torchvision.models : Definitions for popular model architectures, such as AlexNet, VGG, and ResNet and pre-trained models.
-torchvision.transforms : Common image transformations such as random crop, rotations etc.
-torchvision.utils : Useful stuff such as saving tensor (3 x H x W) as image to disk, given a mini-batch creating a grid of images, etc.

```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
```
### Building the Network
We define our own class `class Net(nn.Module)` and we inharite nn.Module which is Base class for all neural network modules. Then we define initialize function `__init__` after we inherite all the functionality of nn.Module in our class `super(Net, self).__init__()`. After that we start building our model.

We'll use 2-D convolutional layers. As activation function we'll choose rectified linear units (ReLUs in short). We use Maxpooling of kernel size 2x2 to reduce channel size into half.

The `forward()` pass defines the way we compute our output using the given layers and functions.

`x.view(-1, 10)` The view method returns a tensor with the same data as the self tensor (which means that the returned tensor has the same number of elements), but with a different shape.
First parameter represent the batch_size in our case batch_size is 128 if you don't know the batch_size pass `-1` tensor.view method will take care of batch_size for you. Second parameter is the column or the number of neurons you want.

`F.log_softmax(x)` log_softmax is an activation function 
- Cross Entropy loss comes in hand with hand with the Softmax layer
- The Softmax layer turns all the class probabilities to values that sum up to 1
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)      #input-28x28  Output-28x28   RF-3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     #input-28x28  Output-28x28   RF-5x5
        self.pool1 = nn.MaxPool2d(2, 2)                  #input-28x28  Output-14x14   RF-10x10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    #input-14x14  Output-14x14   RF-12x12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)   #input-14x14  Output-14x14   RF-14x14
        self.pool2 = nn.MaxPool2d(2, 2)                  #input-14x14  Output-7x7     RF-28x28
        self.conv5 = nn.Conv2d(256, 512, 3)              #input-7x7    Output-5x5     RF-30x30
        self.conv6 = nn.Conv2d(512, 1024, 3)             #input-5x5    Output-3x3     RF-32x32
        self.conv7 = nn.Conv2d(1024, 10, 3)              #input-3x3    Output-1x1     RF-34x34

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
```
### Check for GPU and summerize the model
`from torchsummary import summary`

Torch-summary provides information complementary to what is provided by print(your_model) in PyTorch.  `summary(your_model, input_data)`

`torch.cuda.is_available()` check for the GPU return True if GPU available else return False

`model = Net().to(device)` load model to available device
```python
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```
###Preparing the Dataset
We will be using the popular MNIST database. It is a collection of 70000 handwritten digits split into training and test set of 60000 and 10000 images respectively.

`datasets.MNIST` we are downloading the MNIST dataset for training and testing at path `../data`

Before downloading the data, let us define what are the transformations we want to perform on our data before feeding it into the pipeline. In other words, you can consider it to be some kind of custom edit to are performing to the images so that all the images have the same dimensions and properties. We do it using `torchvision.transforms`.

`transforms.ToTensor()` — converts the image into numbers, that are understandable by the system. It separates the image into three color channels (separate images): red, green & blue. Then it converts the pixels of each image to the brightness of their color between 0 and 255. These values are then scaled down to a range between 0 and 1. The image is now a Torch Tensor.

`transforms.Normalize()` — normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively.

`torch.utils.data.DataLoader` we make Data iterable by loading it to a loader.

`shuffle=True` Shuffle the training data to make it independent of the order  
```python
torch.manual_seed(1)
batch_size = 128 #batch size is the number of images we want to read in one go

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

```
### Training and Test function
`tqdm` which can mean "progress," 

Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you're done!

`model.train()`

By default all the modules are initialized to train mode (self.training = True). Also be aware that some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
Hence we mention in first line of train function i.e `model.train()`
and in first line of test function i.e `model.eval()`

`zero_grad` clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward()

`loss.backward()` computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.

`optimizer.step()` causes the optimizer to take a step based on the gradients of the parameters.

`F.nll_loss` we define the negative log-likelihood loss. It is useful to train a classification problem with C classes. Together the LogSoftmax() and NLLLoss() acts as the cross-entropy loss

```python
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # loading images and labels to available device
        optimizer.zero_grad()
        output = model(data)
        #Calculate loss
        loss = F.nll_loss(output, target)
        #backpropagate the loss 
        loss.backward()
        #Update the weights
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
### Training Model
This is where the actual magic happens. Your neural network iterates over the training set and updates the weights. We make use of `torch.optim` which is a module provided by PyTorch to optimize the model, perform gradient descent and update the weights by back-propagation. Thus in each `epoch` (number of times we iterate over the training set), we will be seeing a gradual decrease in training loss.
```python

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 10):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```
