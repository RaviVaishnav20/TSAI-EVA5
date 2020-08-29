from utils import *
import data_transformations as t

BATCH_SIZE = 128
class DataLoaders():
    def __init__(self):
        pass
    def dataload():

        #device
        cuda = torch.cuda.is_available()

        # Download Dataset
        train_dataset = datasets.MNIST(root='./data', train=True, transform=t.train_transforms, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=t.test_transforms, download=True)

        # Train and Test dataloader
        dataloader_args = dict(batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) if cuda else dict(
            batch_size=64,
            shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

        return train_dataset, test_dataset, train_loader, test_loader
