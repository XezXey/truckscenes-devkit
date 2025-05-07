import numpy as np
import torch as th
from torch.utils.data import Dataset
from truckscenes import TruckScenes
from pyquaternion import Quaternion
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
    trucksc = TruckScenes(version=version, dataroot=dataroot, verbose=True)
    ts_dataset = TruckScenesDataset(trucksc, cfg)
    
    if deterministic:
        ts_dataloader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=cfg.training.batch_size,
            collate_fn=ts_dataset.collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    else:
        ts_dataloader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=cfg.training.batch_size,
            collate_fn=ts_dataset.collate_fn,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    
    return ts_dataloader, ts_dataset
    
    
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
        self.padding_value = -1000
        # self.__getitem__(0)
        
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
        cam_img = np.array(Image.open(os.path.join(self.trucksc.dataroot, cam['filename'])))  # HxWxC
    
        # Load the transformation matrix
        sensor2ego_record = self.trucksc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        cam2ego_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        ego2world = self.trucksc.get('ego_pose', pointsensor['ego_pose_token'])
        
        cam_dict = {
            'sensor2ego': {
                'R': Quaternion(sensor2ego_record['rotation']).rotation_matrix,
                'T': np.array(sensor2ego_record['translation'])
            },
            'ego2global': {
                'R': Quaternion(cam2ego_record['rotation']).rotation_matrix,
                'T': np.array(cam2ego_record['translation'])
            },
            'ego2world': {
                'R': Quaternion(ego2world['rotation']).rotation_matrix,
                'T': np.array(ego2world['translation'])
            }
        }
        # print("[#] pc shape: ", pc.points.shape)
        # print("[#] cam_img shape: ", cam_img.size)
        
        return pc.points.transpose(1, 0), cam_img, cam_dict

    def collate_fn(self, batch):
        
        return {
            'pc': th.rand((3, 10, 10, 10)),
            'img': th.rand((3, 128, 10, 10))
        }
        pc, img, cam_dict = map(list, zip(*batch))
        max_len = max([pc[i].shape[0] for i in range(len(pc))])
        if max_len > self.cfg.training.max_pc_len:
            max_len = self.cfg.training.max_pc_len
            
        #TODO: This can be randomly sampled from start=[0, max_len - len(pc[i])] to start + len(pc[i])
        pc = [th.tensor(pc[i][:max_len, :].copy()) for i in range(len(pc))]
        pc_tensor = th.nn.utils.rnn.pad_sequence(pc, batch_first=True, padding_value=self.padding_value)
        mask_tensor = pc_tensor != self.padding_value
        
        img_tensor = th.stack([th.tensor(img[i].transpose(2, 0, 1).copy()) for i in range(len(img))], dim=0)

        #TODO: Making the cam_dict into a padded tensor
        # 1. Rotation matrix [3x3] -> [BxTx3x3] using np.eye(3)
        # 2. Translation vector [3] -> [BxTx3] using np.zeros(3)
        out_cam_dict = {
            'sensor2ego': {
                'R': [],
                'T': []
            },
            'ego2global': {
                'R': [],
                'T': []
            },
            'ego2world': {
                'R': [],
                'T': []
            }
        }
        # for i in range(len(cam_dict)):
        #     for key in cam_dict[i].keys():
        #         R = cam_dict[i][key]['R'][None, None, ...]   # 1x1x3x3
        #         T = cam_dict[i][key]['T'][None, None, ...]   # 1x1x3
        #         R = np.repeat(R, (1, max_len, 1, 1))
        #         T = np.repeat(T, (1, max_len, 1))
        #         out_cam_dict[key]['R'].append(R)
        #         out_cam_dict[key]['T'].append(T)
            
        # for key in out_cam_dict.keys():
        #     out_cam_dict[key]['R'] = th.from_numpy(np.concatenate(out_cam_dict[key]['R'], axis=0))
        #     out_cam_dict[key]['T'] = th.from_numpy(np.concatenate(out_cam_dict[key]['T'], axis=0))
            # print(key, out_cam_dict[key]['R'].shape, out_cam_dict[key]['T'].shape)
    
        return {
            'pc': th.clone(pc_tensor),
            'img': th.clone(img_tensor),
            'mask': th.clone(mask_tensor),
            # 'cam_dict': out_cam_dict
        }