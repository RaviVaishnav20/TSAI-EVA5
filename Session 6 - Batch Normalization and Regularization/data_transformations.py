# Train phase transformation
from utils import *
train_transforms = transforms.Compose([transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((.1307,), (.3081,)),

                                       ])

# Test phase transformation
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.1307,), (.3081,))
])