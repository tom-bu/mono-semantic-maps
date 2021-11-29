import os
from torch.utils.data import DataLoader, RandomSampler
from .augmentation import AugmentedMapDataset

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from .argoverse.dataset import ArgoverseMapDataset
from .argoverse.splits import TRAIN_LOGS, VAL_LOGS

def build_argoverse_datasets(config):
    print('==> Loading Argoverse dataset...')
    dataroot = os.path.expandvars(config.dataroot)
    
    # Load native argoverse splits
    loaders = {
        # 'train' : ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        # 'val' : ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
        'train' : ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        'val' : ArgoverseTrackingLoader(os.path.join(dataroot, 'train'))
    }


    train_loaders = {
        'train' : ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
    }

    # since we are using a subset of train set as validation, we need to
    # set the identifier as train.
    val_loaders = {
        'val' : ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
    }

    # Create datasets using new argoverse splits
    train_data = ArgoverseMapDataset(train_loaders, config.label_root, 
                                     config.img_size, TRAIN_LOGS)

    print("length of the train data set: {}".format(len(train_data)))
    val_data = ArgoverseMapDataset(val_loaders, config.label_root, 
                                   config.img_size, VAL_LOGS)
    return train_data, val_data


def build_datasets(dataset_name, config):
    return build_argoverse_datasets(config)


def build_trainval_datasets(dataset_name, config):

    # Construct the base dataset
    train_data, val_data = build_datasets(dataset_name, config)

    # Add data augmentation to train dataset
    train_data = AugmentedMapDataset(train_data, config.hflip)

    return train_data, val_data


def build_dataloaders(dataset_name, config):

    # Build training and validation datasets
    train_data, val_data = build_trainval_datasets(dataset_name, config)

    # Create training set dataloader
    # sampler = RandomSampler(train_data, True, config.epoch_size)

    # CHANGED
    sampler = RandomSampler(train_data, True, len(train_data))
    train_loader = DataLoader(train_data, config.batch_size, sampler=sampler,
                              num_workers=config.num_workers)

    
    # Create validation dataloader
    val_loader = DataLoader(val_data, config.batch_size, 
                            num_workers=config.num_workers)
    
    return train_loader, val_loader

    


    

