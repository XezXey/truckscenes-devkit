import numpy as np
import torch as th
from torch.utils.data import Dataset
from truckscenes import TruckScenes
import os
from PIL import Image
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

SENSOR_PAIR_ID = {
    'RADAR_LEFT_FRONT': 'CAMERA_LEFT_FRONT',
    'RADAR_RIGHT_FRONT': 'CAMERA_RIGHT_FRONT',
}

def get_truckscenes_dataset(cfg, deterministic=False):
    """
    Load the TruckScenes dataset.
    
    Args:
        root_dir (str): Path to the root directory of the dataset.
        version (str): Version of the dataset to load.
        
    
    Returns:
        TruckScenes: Loaded TruckScenes dataset.
    """
    
    version = cfg.dataset.version
    dataroot = cfg.dataset.dataroot
    sample_token = cfg.dataset.sample_token
    trucksc = TruckScenes(version=version, dataroot=dataroot, verbose=True)
    ts_dataset = TruckScenesDataset(trucksc, cfg)
    
    print(trucksc.scene)
    if deterministic:
        ts_loader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        ts_loader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    
    return ts_loader, ts_dataset
    exit()
    
    
class TruckScenesDataset(Dataset):
    """
    Custom dataset class for TruckScenes.
    
    Args:
        root_dir (str): Path to the root directory of the dataset.
        version (str): Version of the dataset to load.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, trucksc, cfg):
        self.trucksc = trucksc
        self.cfg = cfg
        self.scene = trucksc.scene  # List of scenes
        self.sample_token = cfg.dataset.sample_token
        self.radar_position = cfg.dataset.radar_position[0] # Assuming only one radar position is used
        self.__getitem__(0)
        
    def __len__(self):
        return len(self.scene)
    
    def __getitem__(self, idx):
        scene = self.scene[idx]
        sample_record = self.trucksc.get('sample', scene[self.sample_token])
        pointsensor_token = sample_record['data'][self.radar_position]
        camera_token = sample_record['data'][SENSOR_PAIR_ID[self.radar_position]]
        cam = self.trucksc.get('sample_data', camera_token)
        pointsensor = self.trucksc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(self.trucksc.dataroot, pointsensor['filename'])
        
        # Load the point cloud
        pc = RadarPointCloud.from_file(pcl_path)    # 7xN; 7 is x, y, z, vx, vy, vz, intensity
        # Load the camera image
        cam_img = Image.open(os.path.join(self.trucksc.dataroot, cam['filename']))  # HxWxC
        