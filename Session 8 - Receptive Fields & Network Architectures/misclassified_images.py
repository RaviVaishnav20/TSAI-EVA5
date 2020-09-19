from utils import *
import matplotlib.gridspec as gridspec
# view and save misclassified images
import math

PATH = "/images/"


class MissclassifiedImages():
    def __init__(self):
        pass

    def show_save_misclassified_images(model, device, test_loader, classes, img_mean, img_std, img_name, PATH,max_misclassified_imgs):
        fig = plt.figure(figsize=(16, 10))
        shown_batch = 0
        index = 0
        num_rows = round(max_misclassified_imgs/5)
        print(num_rows)
        with torch.no_grad():
            correct = 0
            total = 0
            wrong_pred_len = 0

            for images, labels in test_loader:
                if wrong_pred_len > max_misclassified_imgs:
                    break

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # the output is of shape (4,2)
                _, predicted = torch.max(outputs.data, 1)  # the prediction is of shape (4) ------------> [0, 0, 0, 1]
                wrong_pred = predicted[predicted.eq(labels.view_as(predicted)) == False]
                ground_truth = labels[labels.eq(predicted.view_as(labels)) == False]
                wrong_pred_len += wrong_pred.size(0)

                spec2 = gridspec.GridSpec(ncols=5, nrows=num_rows, figure=fig)
                for i in range(wrong_pred.size(0)):
                    if index >= max_misclassified_imgs:
                        break
                    ax = fig.add_subplot(spec2[index])
                    index += 1
                    ax.axis('off')
                    ax.set_title("Prediction Label: {},\n Ground Truth: {}".format(classes[wrong_pred[i]],
                                                                                    classes[ground_truth[i]]))
                    ax.figure.tight_layout(pad=1.0)
                    input_img = images.cpu().data[i]  # Get the tesnsor of the image and put it to cpu
                    inp = input_img.numpy().transpose((1, 2, 0))  # If we have tensor of shape (2, 3, 4) ----> it becomes(3, 4, 2)
                    mean = np.array(img_mean)

                    std = np.array(img_std)
                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
                    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
                    plt.imshow(inp)
                plt.savefig(PATH + str(img_name) + ".png")