import torch
from torch.utils.data import Dataset
import random

class AugmentedMapDataset(Dataset):

    def __init__(self, dataset, hflip=True):
        self.dataset = dataset
        self.hflip = hflip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, calib, labels, mask, city_SE3_egovehicle = self.dataset[index]

        # Apply data augmentation
        if self.hflip:
            image, labels, mask = random_hflip(image, labels, mask)

        return image, calib, labels, mask

    
def random_hflip(image, labels, mask):
    coin = random.randint(0,1)
    if coin:
        image = torch.flip(image, (-1,))
        labels = torch.flip(labels.int(), (-1,)).bool()
        mask = torch.flip(mask.int(), (-1,)).bool()
    return image, labels, mask