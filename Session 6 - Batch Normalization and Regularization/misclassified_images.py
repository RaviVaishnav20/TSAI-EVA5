from utils import *
# view and save misclassified images
import math

PATH = "/images/"

class MissclassifiedImages():
    def __init__(self):
        pass
    def show_save_misclassified_images(model,device, test_loader, name="fig",PATH="/images/", max_misclassified_imgs=25):
        cols = 5
        rows = math.ceil(max_misclassified_imgs / cols)

        with torch.no_grad():
            ind = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                misclassified_imgs_pred = pred[pred.eq(target.view_as(pred)) == False]

                misclassified_imgs_data = data[pred.eq(target.view_as(pred)) == False]
                misclassified_imgs_target = target[(pred.eq(target.view_as(pred)) == False).view_as(target)]
                if ind == 0:
                    example_data, example_targets, example_preds = misclassified_imgs_data, misclassified_imgs_target, misclassified_imgs_pred
                elif example_data.shape[0] < max_misclassified_imgs:
                    example_data = torch.cat([example_data, misclassified_imgs_data], dim=0)
                    example_targets = torch.cat([example_targets, misclassified_imgs_target], dim=0)
                    example_preds = torch.cat([example_preds, misclassified_imgs_pred], dim=0)
                else:
                    break
                ind += 1
            example_data, example_targets, example_preds = example_data[:max_misclassified_imgs], example_targets[
                                                                                              :max_misclassified_imgs], example_preds[
                                                                                                                        :max_misclassified_imgs]
        fig = plt.figure(figsize=(20, 10))
        for i in range(max_misclassified_imgs):
            plt.subplot(rows, cols, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i].cpu().numpy(), cmap='gray', interpolation='none')
            plt.title(f"{i + 1}) Ground Truth: {example_targets[i]},\n Prediction: {example_preds[i]}")
            plt.xticks([])
            plt.yticks([])
        plt.savefig(PATH + str(name) + ".png")