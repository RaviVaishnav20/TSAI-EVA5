from utils import *

# test_losses = []
# test_accuracy = []


class TestModel():
    def __init__(self):
        pass

    def test(model, device, test_loader, lossFn):
        model.eval()

        test_acc = 0
        correct = 0
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # get samples
                images, labels = images.to(device), labels.to(device)
                # Predict
                y_pred = model(images)
                # Calculate loss and sum all the loss for one batch
                loss = lossFn(y_pred, labels)
                test_loss +=loss.sum().item()
                pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                # check how many predictions are correct
                if device == "cuda":
                    correct += pred.cpu().eq(labels.cpu().view_as(pred)).sum().item()
                else:
                    correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)  # loss per epoch
        #test_losses.append(test_loss)

        test_acc = 100. * correct / len(test_loader.dataset)

        print(
            f"\nTest set: Average Loss= {test_loss :0.4f} Batch_id= {batch_idx} Accuracy= {correct}/{len(test_loader.dataset)} ({test_acc:0.2f}%)\n")
        #test_accuracy.append(test_acc)

        return test_loss, test_acc
