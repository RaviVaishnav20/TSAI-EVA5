from utils import *
def getPlottedGraph(epochs,train_losses, train_accuracy, test_losses, test_accuracy,name="los_fig", PATH="./"):

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


def plot_lr(lrs, moms=None,name="lr_fig", PATH="./" ):
    "Plot learning rate, `show_moms` to include momentum"
    iterations = list(range(len(lrs)))

    plt.figure(figsize=(14, 10))
    plt.subplot(111)
    plt.title("Learning Rate")
    plt.plot(iterations, lrs, color='r', label="test accuracy")
    
    plt.savefig(PATH + str(name) + ".png")
    plt.show()

    
    # if moms!= None:
    #     fig, axs = plt.subplots(1, 2, figsize=(18, 10))
    #     axs[0].plot(iterations, lrs)
    #     axs[0].set_title("Learning Rate")
    #     axs[1].plot(iterations, moms)
    #     axs[1].set_title("Momentum")
    #     plt.savefig(PATH + str(name) + ".png")
    #     plt.show()
    # else:
    #     fig, axs = plt.subplots(1, 1, figsize=(14, 10))
    #     axs[0].plot(iterations, lrs)
    #     axs[0].set_title("Learning Rate")
    #     plt.savefig(PATH + str(name) + ".png")
    #     plt.show()


    

    

