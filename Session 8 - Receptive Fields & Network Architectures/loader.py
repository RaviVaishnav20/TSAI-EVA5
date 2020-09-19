from utils import *

class Loader():
    def __init__(self):
        pass

    def getDataLoader(dataset_name, trainSet_dict, trainLoad_dict):
        return torch.utils.data.DataLoader(dataset_name(**trainSet_dict), **trainLoad_dict)
