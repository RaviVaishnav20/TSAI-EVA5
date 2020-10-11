from utils import *
from one_cycle_policy import *
#from regularization import Regularization as reg
from test import TestModel

def trainNetwork(model,device, trainloader, testloader, optimizer, criterion, epochs, lr_max, moms, div_factor, pct_start,
                 MODEL_PATH='./', L1_regularization=None, WEIGHT_DECAY=1e-2):
    filename = MODEL_PATH + "S11_model.pth"
    ocs = OneCycleScheduler(model, trainloader, optimizer, criterion, lr_max, moms, div_factor, pct_start)
    # For Graph Learning Rate or Momentum
    lr_list = []
    moms_list = []
    # For Graph loss vs accuracy
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    for epoch in range(epochs):
        LR, MOMENTUM = ocs.on_train_begain(epochs)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        model.train()
        train_acc = 0
        correct = 0
        processed = 0

        pbar = tqdm(trainloader, position=0, leave=True)

        for batch_idx, (images, labels) in enumerate(pbar):
            # get samples
            images, labels = images.to(device), labels.to(device)
            # Init
            optimizer.zero_grad()
            # Predict
            y_pred = model(images)
            # Calculate loss
            loss = criterion(y_pred, labels)
            if L1_regularization:
                loss = L1_regularization.l1_reg(model, loss, 0.0001)
            # train_losses.append(loss)

            # Backpropagation
            loss.backward()

            # update weights
            optimizer.step()

            LR, MOMENTUM = ocs.on_batch_end()
            lr_list.append(LR)
            moms_list.append(MOMENTUM)

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # check how many predictions are correct
            if device == "cuda":
                correct += pred.cpu().eq(labels.cpu().view_as(pred)).sum().item()
            else:
                correct += pred.eq(labels.view_as(pred)).sum().item()

            processed += len(images)  # 128 + 128 +128 ......till 60000 images

            train_acc = 100 * correct / processed

            pbar.set_description(
                desc=f": EPOCH= {epoch} Loss= {loss.item() :0.4f} Batch_id= {batch_idx} Accuracy= {train_acc:0.2f}")
            # train_accuracy.append(train_acc)

        train_losses.append(loss.item())
        train_accuracy.append(train_acc)

        test_loss, test_acc = TestModel.test(model, device, testloader, criterion)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)

        # Save Model
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, filename)

    return lr_list, moms_list, train_losses, train_accuracy, test_losses, test_accuracy, optimizer