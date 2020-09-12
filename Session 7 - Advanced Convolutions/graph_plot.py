from utils import *
def getPlottedGraph(epochs,train_losses, train_accuracy, test_losses, test_accuracy,name=None, PATH=None):

    x_epochs = np.arange(0, epochs, 1)

    plt.figure(figsize=(25,15))
    plt.subplot(221)
    plt.title("Training Loss")
    plt.plot(x_epochs, train_losses, color='r', label="train loss")
    plt.legend()

    plt.subplot(222)
    plt.title("Test Loss")
    plt.plot(x_epochs, test_losses, color='g', label="test loss")
    plt.legend()

    plt.subplot(223)
    plt.title("Training Accuracy")
    plt.plot(x_epochs, train_accuracy, color='c', label="train accuracy")
    plt.legend()

    plt.subplot(224)
    plt.title("Test Accuracy")
    plt.plot(x_epochs, test_accuracy, color='r', label="test accuracy")
    plt.legend()

    plt.savefig(PATH + str(name) + ".png")

    plt.show()


