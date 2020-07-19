## Neural Architecture

### Importing Libraries 

```from __future__ import print_function```

First of all, from __future__ import print_function needs to be the first line of code in your script (aside from some exceptions mentioned below). Second of all, as other answers have said, you have to use print as a function now. That's the whole point of from __future__ import print_function; to bring the print function from Python 3 into Python 2.6+

```import torch```

Import torch to work with PyTorch and perform the operation

```import torch.nn as nn```

torch.nn provide us many more classes and modules to implement and train the neural network.

```import torch.nn.functional as F```

torch.nn.functional is give access to call any function inside torch.nn

**Note:**

nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
nn.functional.relu on the other side is just the functional API call to the relu function, so that you can add it e.g. in your forward method yourself.

Generally speaking it might depend on your coding style if you prefer modules for the activations or the functional calls

```import torch.optim as optim```

torch.optim is a package implementing various optimization algorithms. 

```from torchvision import datasets, transforms```

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
