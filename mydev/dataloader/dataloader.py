import numpy as np
import torch as th
from torch.utils.data import Dataset
from truckscenes import TruckScenes
import os

def get_truckscenes_dataset(root_dir, version):
    """
    Load the TruckScenes dataset.
    
    Args:
        root_dir (str): Path to the root directory of the dataset.
        version (str): Version of the dataset to load.
        
    
    Returns:
        TruckScenes: Loaded TruckScenes dataset.
    """
    
    trucksc = TruckScenes(version=version, dataroot=root_dir, verbose=True)
    print(trucksc.scene)
    exit()
    
    
class TruckScenesDataset(Dataset):
    """
    Custom dataset class for TruckScenes.
    
    Args:
        root_dir (str): Path to the root directory of the dataset.
        version (str): Version of the dataset to load.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, trucksc):
        self.trucksc = trucksc