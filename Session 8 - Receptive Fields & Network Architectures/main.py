from utils import *
from image_transform import ImageTransform
from loader import Loader
from resnet import *
from graph_plot import *
# from model7 import Net, SeparableConv2d
from train import TrainModel
from test import TestModel
from misclassified_images import MissclassifiedImages as ms

# For Graph
train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

# Hyper parameters
LAMBDA1=1e-5
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0001    #
EPOCHS = 50
img_mean = (0.4914, 0.4822, 0.4465)
img_std = (0.2023, 0.1994, 0.2010)
# Transforms
transform_train = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(img_mean, img_std),
]

transform_test = [
    transforms.ToTensor(),
    transforms.Normalize(img_mean,img_std),
]

t_transform_train = ImageTransform.transform(transform_train)
t_transform_test = ImageTransform.transform(transform_test)

# Dataset and DataLoader arguments
dataset_name = torchvision.datasets.CIFAR10
trainSet_dict = dict(root='./data', train=True, download=True, transform=t_transform_train)
trainLoad_dict = dict(batch_size=32, shuffle=True, num_workers=4)
testSet_dict = dict(root='./data', train=False, download=True, transform=t_transform_test)
testLoad_dict = dict(batch_size=32, shuffle=False, num_workers=4)

IMAGE_PATH = "images/"
MODEL_PATH = "model/"
def main():
    # Device
    SEED = 1
    cuda = torch.cuda.is_available()
    print("Cuda is available ?", cuda)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    device = torch.device("cuda" if cuda else "cpu")

    # Create Train and Test Loader
    trainloader = Loader.getDataLoader(dataset_name, trainSet_dict, trainLoad_dict)
    testloader = Loader.getDataLoader(dataset_name, testSet_dict, testLoad_dict)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=13, gamma=0.1)

    # Start training
    for epoch in range(1):
        train_loss, train_acc = TrainModel.train(model, device, trainloader, criterion, optimizer, epoch)
        # scheduler.step()
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        test_loss, test_acc = TestModel.test(model, device, testloader, criterion)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)

    # Plot and Save Graph
    getPlottedGraph(EPOCHS, train_losses, train_accuracy, test_losses, test_accuracy,name="cifar_10_plot_using_resnet18_v3", PATH=IMAGE_PATH)

    # Save Models
    torch.save(model.state_dict(), MODEL_PATH+"model8_v3.pth")

    #misclassified images
    ms.show_save_misclassified_images(model, device, testloader, classes, list(img_mean), list(img_std),
                                      name="fig_cifar10_v1", PATH=IMAGE_PATH,
                                      max_misclassified_imgs=25)


if __name__ == "__main__":
    main()
