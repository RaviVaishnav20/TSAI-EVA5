from utils import *
from regularization import Regularization as reg
from dataloaders import DataLoaders
from model_with_bn import Net as bn_model
from model_with_gbn import Net as gbn_model
from train import TrainModel
from test import TestModel
from graph_plot import getPlottedGraph
from misclassified_images import MissclassifiedImages as MI

# Train and Test model plot
l1_train_losses = []
l1_train_accuracy = []
l2_train_losses = []
l2_train_accuracy = []
l1_l2_train_losses = []
l1_l2_train_accuracy = []
gbn_train_losses = []
gbn_train_accuracy = []
gbn_l2_train_losses = []
gbn_l2_train_accuracy = []
gbn_l1_l2_train_losses = []
gbn_l1_l2_train_accuracy = []

l1_test_losses = []
l1_test_accuracy = []
l2_test_losses = []
l2_test_accuracy = []
l1_l2_test_losses = []
l1_l2_test_accuracy = []
gbn_test_losses = []
gbn_test_accuracy = []
gbn_l2_test_losses = []
gbn_l2_test_accuracy = []
gbn_l1_l2_test_losses = []
gbn_l1_l2_test_accuracy = []

IMAGE_PATH = "/images/"
MODEL_PATH = "/model/"
def main():

    # Hyper parameters
    EPOCHS = 2


    # For reproducibility
    SEED = 1
    # Check for CUDA?
    cuda = torch.cuda.is_available()
    print("Cuda is available ?", cuda)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    train_dataset, test_dataset, train_loader, test_loader = DataLoaders.dataload()

    device = torch.device("cuda" if cuda else "cpu")

    # Summary
    # summary(model, input_size=(1, 28, 28))

    # Optimizer
    model1 = bn_model().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    scheduler1 = StepLR(optimizer1, step_size=7, gamma=0.1)

    model2 = gbn_model().to(device)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    scheduler2 = StepLR(optimizer2, step_size=7, gamma=0.1)

    for epoch in range(EPOCHS):
        # With L1
        l1_train_loss, l1_train_acc = TrainModel.train(model1, device, train_loader, optimizer1, epoch,L1_regularization=reg,m_type="L1")
        l1_train_losses.append(l1_train_loss)
        l1_train_accuracy.append(l1_train_acc)
        #scheduler1.step_size = 23
        scheduler1.step()
        l1_test_loss, l1_test_acc = TestModel.test(model1, device, test_loader)
        l1_test_losses.append(l1_test_loss)
        l1_test_accuracy.append(l1_test_acc)

        # With L2
        optimizer1.param_groups[0]['weight_decay'] = 0.0001
        l2_train_loss, l2_train_acc = TrainModel.train(model1, device, train_loader, optimizer1, epoch,m_type="L2")
        l2_train_losses.append(l2_train_loss)
        l2_train_accuracy.append(l2_train_acc)
        #scheduler1.step_size = 3
        scheduler1.step()
        l2_test_loss, l2_test_acc = TestModel.test(model1, device, test_loader)
        l2_test_losses.append(l2_test_loss)
        l2_test_accuracy.append(l2_test_acc)

        # With L1 and L2
        optimizer1.param_groups[0]['weight_decay'] = 0.0001
        l1_l2_train_loss, l1_l2_train_acc = TrainModel.train(model1, device, train_loader, optimizer1, epoch, L1_regularization=reg,m_type="L1&L2")
        l1_l2_train_losses.append(l1_l2_train_loss)
        l1_l2_train_accuracy.append(l1_l2_train_acc)
       # scheduler1.step_size = 19
        scheduler1.step()
        l1_l2_test_loss, l1_l2_test_acc = TestModel.test(model1, device, test_loader)
        l1_l2_test_losses.append(l1_l2_test_loss)
        l1_l2_test_accuracy.append(l1_l2_test_acc)

        # With GBN
        gbn_train_loss, gbn_train_acc = TrainModel.train(model2, device, train_loader, optimizer2, epoch, m_type="GBN")
        gbn_train_losses.append(gbn_train_loss)
        gbn_train_accuracy.append(gbn_train_acc)
       # scheduler2.step_size = 11
        scheduler2.step()
        gbn_test_loss, gbn_test_acc = TestModel.test(model2, device, test_loader)
        gbn_test_losses.append(gbn_test_loss)
        gbn_test_accuracy.append(gbn_test_acc)

        # GBN With L2
        optimizer2.param_groups[0]['weight_decay'] = 0.0001
        gbn_l2_train_loss, gbn_l2_train_acc = TrainModel.train(model2, device, train_loader, optimizer2, epoch,  m_type="GBN&L2")
        gbn_l2_train_losses.append(gbn_l2_train_loss)
        gbn_l2_train_accuracy.append(gbn_l2_train_acc)
       # scheduler2.step_size = 6
        scheduler2.step()
        gbn_l2_test_loss, gbn_l2_test_acc = TestModel.test(model2, device, test_loader)
        gbn_l2_test_losses.append(gbn_l2_test_loss)
        gbn_l2_test_accuracy.append(gbn_l2_test_acc)

        # GBN With L1 and L2
        optimizer2.param_groups[0]['weight_decay'] = 0.0001
        gbn_l1_l2_train_loss, gbn_l1_l2_train_acc = TrainModel.train(model2, device, train_loader, optimizer2, epoch, L1_regularization=reg, m_type="GBN&L1&L2")
        gbn_l1_l2_train_losses.append(gbn_l1_l2_train_loss)
        gbn_l1_l2_train_accuracy.append(gbn_l1_l2_train_acc)
       # scheduler2.step_size = 21
        scheduler2.step()
        gbn_l1_l2_test_loss, gbn_l1_l2_test_acc = TestModel.test(model2, device, test_loader)
        gbn_l1_l2_test_losses.append(gbn_l1_l2_test_loss)
        gbn_l1_l2_test_accuracy.append(gbn_l1_l2_test_acc)

    #Save Models
    #PATH = "/content/drive/My Drive/Lab/Loss_and_accuracy_plot.png"
    torch.save(model1, MODEL_PATH)
    torch.save(model2, MODEL_PATH)

    #Plot and save graph of losses and accuracy
    getPlottedGraph(EPOCHS, l1_train_losses, l1_train_accuracy, l1_test_losses, l1_test_accuracy,
                    l2_train_losses, l2_train_accuracy,l2_test_losses, l2_test_accuracy,
                    l1_l2_train_losses, l1_l2_train_accuracy,l1_l2_test_losses, l1_l2_test_accuracy,
                    gbn_train_losses, gbn_train_accuracy, gbn_test_losses, gbn_test_accuracy,
                    gbn_l2_train_losses, gbn_l2_train_accuracy, gbn_l2_test_losses, gbn_l2_test_accuracy,
                    gbn_l1_l2_train_losses, gbn_l1_l2_train_accuracy, gbn_l1_l2_test_losses, gbn_l1_l2_test_accuracy,name="plot", PATH=IMAGE_PATH)

    #Save misclassified images
    MI.show_save_misclassified_images(model2, test_loader, name="fig1",PATH=IMAGE_PATH, max_misclassified_imgs=25)
    MI.show_save_misclassified_images(model2, test_loader, name="fig2",PATH=IMAGE_PATH, max_misclassified_imgs=25)
if __name__ == "__main__":
    main()
