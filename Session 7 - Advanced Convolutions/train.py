from utils import *
#from regularization import Regularization as reg

# train_losses = []
# train_accuracy = []

class TrainModel():
    def __init__(self):
        pass


    def train(model, device, train_loader, lossFn, optimizer, epoch, L1_regularization=None, m_type=""):
        model.train()
        train_acc = 0
        correct = 0
        processed = 0

        pbar = tqdm(train_loader, position=0, leave=True)

        for batch_idx, (images, labels) in enumerate(pbar):
            # get samples
            images, labels = images.to(device), labels.to(device)
            # Init
            optimizer.zero_grad()
            # Predict
            y_pred = model(images)
            # Calculate loss
            loss = lossFn(y_pred, labels)
            if L1_regularization:
                loss = L1_regularization.l1_reg(model, loss, 0.0001)
            # train_losses.append(loss)

            # Backpropagation
            loss.backward()

            # update weights
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # check how many predictions are correct
            if device == "cuda":
                correct += pred.cpu().eq(labels.cpu().view_as(pred)).sum().item()
            else:
                correct += pred.eq(labels.view_as(pred)).sum().item()

            processed += len(images)  # 128 + 128 +128 ......till 60000 images

            train_acc = 100 * correct / processed

            pbar.set_description(
                desc=f"{m_type}: EPOCH= {epoch} Loss= {loss.item() :0.4f} Batch_id= {batch_idx} Accuracy= {train_acc:0.2f}")
            # train_accuracy.append(train_acc)

        return loss.item(), train_acc