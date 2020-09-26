from utils import *
from loader import Loader
from resnet import *
from graph_plot import *
from albumentation_transform import AlbumentationTransformations
from train import TrainModel
from test import TestModel
from display_images import *
from albumentations import *
from albumentations.pytorch.transforms import ToTensor
from grad_cam import *

# ----------------------------------------------------------------- DEVICE ---------------------------------------------------------------------->

SEED = 1
cuda = torch.cuda.is_available()
# print("Cuda is available ?", cuda)
torch.manual_seed(SEED)
if cuda:
    torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if cuda else "cpu")

# --------------------------------------------------------- DEFAULTS PATH AND HYPERPARAMETERS ----------------------------------------------------->

IMAGE_PATH = "./visualization/"
MODEL_PATH = "./model_weights/"

# For Graph
train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

# Hyper parameters
LAMBDA1 = 1e-5
LR = 0.04
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001  #
EPOCHS = 10
img_mean = (0.4914, 0.4822, 0.4465)
img_std = (0.2023, 0.1994, 0.2010)

# --------------------------------------------------------------- DATA AUGMENTATION AND DATA LOADERS --------------------------------------------->

# Transforms
means = np.array(img_mean)
stdevs = np.array(img_std)
patch_size = 28

# Define Train transforms and Test transforms
# Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=0.75),
transform_train = [

    RGBShift(),
    RandomRotate90(),
    HorizontalFlip(p=0.5),
    Normalize(mean=means, std=stdevs),
    ToTensor()
]

transform_test = [Normalize(mean=means, std=stdevs),
                  ToTensor()
                  ]

# Create Train transforms and Test transforms
t_transform_train = AlbumentationTransformations(transform_train)
t_transform_test = AlbumentationTransformations(transform_test)

# Dataset and DataLoader arguments
dataset_name = torchvision.datasets.CIFAR10
trainSet_dict = dict(root='./data', train=True, download=True, transform=t_transform_train)
trainLoad_dict = dict(batch_size=64, shuffle=True, num_workers=4)
testSet_dict = dict(root='./data', train=False, download=True, transform=t_transform_test)
testLoad_dict = dict(batch_size=32, shuffle=False, num_workers=4)

# ----------------------------------------------------------- LOSS FUNCTION, OPTIMIZER AND SCHEDULER ----------------------------------------------->

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
model = ResNet18().to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=23, gamma=0.4)

# ----------------------------------------------------------------------- START TRAINING ----------------------------------------------------------->

filename = MODEL_PATH + "S9_model_wt.pth"


def main():
    # Create Train Loader and Test Loader
    trainloader = Loader.getDataLoader(dataset_name, trainSet_dict, trainLoad_dict)
    testloader = Loader.getDataLoader(dataset_name, testSet_dict, testLoad_dict)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Start training
    for epoch in range(EPOCHS):
        #Train
        train_loss, train_acc = TrainModel.train(model, device, trainloader, criterion, optimizer, epoch)
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        #Test
        test_loss, test_acc = TestModel.test(model, device, testloader, criterion)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)

        #Save model
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, filename)


    # Plot and Save Graph
    getPlottedGraph(EPOCHS, train_losses, train_accuracy, test_losses, test_accuracy, name="S9_plot_final",
                        PATH=MODEL_PATH)
    # Show and Save correct classified images
    show_save_correctly_classified_images(model, testloader, device, IMAGE_PATH, name="correct_classified_imgs",
                                          max_correctly_classified_images_imgs=25, labels_list=classes)
    # Show and Save misclassified images
    show_save_misclassified_images(model, testloader, device, IMAGE_PATH, name="misclassified_imgs",
                                   max_misclassified_imgs=25, labels_list=classes)
    # Visualize Activation Map
    misclassified_imgs, correctly_classified_images = classify_images(model, testloader, device, 5)
    layers_list = ["layer1", "layer2", "layer3", "layer4"]
    display_gradcam = VisualizeCam(model, classes, layers_list)
    correct_pred_imgs = []
    for i in range(len(correctly_classified_images)):
        correct_pred_imgs.append(torch.as_tensor(correctly_classified_images[i]["img"]))
    display_gradcam(torch.stack(correct_pred_imgs), layers_list, PATH="./" + str("visualization"), metric="correct")




if __name__ == "__main__":
    main()
