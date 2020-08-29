from utils import *;

def getPlottedGraph(epochs,
                    l1_train_losses, l1_train_accuracy, l1_test_losses, l1_test_accuracy,
                    l2_train_losses, l2_train_accuracy,l2_test_losses, l2_test_accuracy,
                    l1_l2_train_losses, l1_l2_train_accuracy,l1_l2_test_losses, l1_l2_test_accuracy,
                    gbn_train_losses, gbn_train_accuracy, gbn_test_losses, gbn_test_accuracy,
                    gbn_l2_train_losses, gbn_l2_train_accuracy, gbn_l2_test_losses, gbn_l2_test_accuracy,
                    gbn_l1_l2_train_losses, gbn_l1_l2_train_accuracy, gbn_l1_l2_test_losses, gbn_l1_l2_test_accuracy,name="plot", PATH="/images/"):

    x_epochs = np.arange(0, epochs, 1)

    plt.figure(figsize=(25,15))
    plt.subplot(221)
    plt.title("Training Loss")
    plt.plot(x_epochs, l1_train_losses, color='r', label="L1 train loss")
    plt.plot(x_epochs, l2_train_losses, color='g', label="L2 train loss")
    plt.plot(x_epochs, l1_l2_train_losses, color='b', label="L1&L2 train loss")
    plt.plot(x_epochs, gbn_train_losses, color='c', label="GBN train loss")
    plt.plot(x_epochs, gbn_l2_train_losses, color='y', label="GBN&L2 train loss")
    plt.plot(x_epochs, gbn_l1_l2_train_losses, color='m', label="GBN&L1&L2 train loss")
    plt.legend()

    plt.subplot(222)
    plt.title("Test Loss")
    plt.plot(x_epochs, l1_test_losses, color='r', label="L1 test loss")
    plt.plot(x_epochs, l2_test_losses, color='g', label="L2 test loss")
    plt.plot(x_epochs, l1_l2_test_losses, color='b', label="L1&L2 test loss")
    plt.plot(x_epochs, gbn_test_losses, color='c', label="GBN train loss")
    plt.plot(x_epochs, gbn_l2_test_losses, color='y', label="GBN&L2 test loss")
    plt.plot(x_epochs, gbn_l1_l2_test_losses, color='m', label="GBN&L1&L2 train loss")
    plt.legend()

    plt.subplot(223)
    plt.title("Training Accuracy")
    plt.plot(x_epochs, l1_train_accuracy, color='r', label="L1 train accuracy")
    plt.plot(x_epochs, l2_train_accuracy, color='g', label="L2 train accuracy")
    plt.plot(x_epochs, l1_l2_train_accuracy, color='b', label="L1&L2 train accuracy")
    plt.plot(x_epochs, gbn_train_accuracy, color='c', label="GBN train accuracy")
    plt.plot(x_epochs, gbn_l2_train_accuracy, color='y', label="GBN&L2 train accuracy")
    plt.plot(x_epochs, gbn_l1_l2_train_accuracy, color='m', label="GBN&L1&L2 train accuracy")
    plt.legend()

    plt.subplot(224)
    plt.title("Test Accuracy")
    plt.plot(x_epochs, l1_test_accuracy, color='r', label="L1 test accuracy")
    plt.plot(x_epochs, l2_test_accuracy, color='g', label="L2 test accuracy")
    plt.plot(x_epochs, l1_l2_test_accuracy, color='b', label="L1&L2 test accuracy")
    plt.plot(x_epochs, gbn_test_accuracy, color='c', label="GBN train accuracy")
    plt.plot(x_epochs, gbn_l2_test_accuracy, color='y', label="GBN&L2 train accuracy")
    plt.plot(x_epochs, gbn_l1_l2_test_accuracy, color='m', label="GBN&L1&L2 train accuracy")
    plt.legend()
    plt.savefig(PATH + str(name) + ".png")

    plt.show()


