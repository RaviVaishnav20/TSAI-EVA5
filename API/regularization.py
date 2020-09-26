from utils import  *

class Regularization():
    def __init__(self):
        pass

    def l1_reg(model, loss, lambda_value):
        l1 = 0
        #print(f"Previous loss: {loss}")
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            loss = loss + l1 * lambda_value
       #     print(f"Loss after L1: {loss} l1 value: {l1}")
        return loss
