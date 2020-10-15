from lr_finder import LRFinder
import math

def get_LR(model,trainloader, optimizer, criterion, device,testloader=None):

    # print("########## Tweaked version from fastai ###########")
    # lr_find = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_find.range_test(trainloader, end_lr=100, num_iter=100)
    # best_lr=lr_find.plot()  # to inspect the loss-learning rate graph
    # lr_find.reset()
    # return best_lr

    # print("########## Tweaked version from fastai ###########")
    # lr_find = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_find.range_test(trainloader, end_lr=1, num_iter=100)
    # lr_find.plot() # to inspect the loss-learning rate graph
    # lr_find.reset()
    # for index in range(len(lr_find.history['loss'])):
    #   item = lr_find.history['loss'][index]
    #   if item == lr_find.best_loss:
    #     min_val_index = index
    #     print(f"{min_val_index}")
    #
    # lr_find.plot(show_lr=lr_find.history['lr'][75])
    # lr_find.plot(show_lr=lr_find.history['lr'][min_val_index])
    #
    # val_index = 75
    # mid_val_index = math.floor((val_index + min_val_index)/2)
    # show_lr=[{'data': lr_find.history['lr'][val_index], 'linestyle': 'dashed'}, {'data': lr_find.history['lr'][mid_val_index], 'linestyle': 'solid'}, {'data': lr_find.history['lr'][min_val_index], 'linestyle': 'dashed'}]
    # # lr_find.plot_best_lr(skip_start=10, skip_end=5, log_lr=True, show_lr=show_lr, ax=None)
    #
    # best_lr = lr_find.history['lr'][mid_val_index]
    # print(f"LR to be used: {best_lr}")
    #
    # return best_lr

    print("########## Leslie Smith's approach ###########")
    lr_find = LRFinder(model, optimizer, criterion, device="cuda")
    lr_find.range_test(trainloader,val_loader=testloader, end_lr=1, num_iter=100, step_mode="linear")
    best_lr=lr_find.plot(log_lr=False)
    lr_find.reset()
    return best_lr
