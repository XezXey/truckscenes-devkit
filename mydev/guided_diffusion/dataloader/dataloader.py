import numpy as np
import torch as th
from torch.utils.data import Dataset
from truckscenes import TruckScenes
from pyquaternion import Quaternion
import os, json, datetime
from PIL import Image
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

SENSOR_PAIR_ID = {
    'RADAR_LEFT_FRONT': 'CAMERA_LEFT_FRONT',
    'RADAR_RIGHT_FRONT': 'CAMERA_RIGHT_FRONT',
}

def get_scene_token(trucksc, scene):
    print("=" * 100)
    print("[#] scene:", scene)
    print(scene.keys())
    print("[#] First_sample_token: ", scene['first_sample_token'])
    print("[#] Last_sample_token: ", scene['last_sample_token'])
    current_token = trucksc.get('sample', scene['first_sample_token'])
    samples_token = [current_token['token']]
    timestamps = [current_token['timestamp']]
    count = 1

    assert current_token['prev'] == '', "[#] First sample should not have a previous sample."

    while current_token['next'] != '':
        count += 1
        current_token = trucksc.get('sample', current_token['next'])
        samples_token.append(current_token['token'])
        timestamps.append(current_token['timestamp'])
        assert current_token['scene_token'] == scene['token'], "[#] Sample should belong to the same scene."
    
    assert trucksc.get('sample', samples_token[-1])['next'] == '', "[#] Last sample should not have a next sample."
    print("[#] #count:", count)
    print("[#] #nbr_samples: ", scene['nbr_samples'])
    assert count == scene['nbr_samples'], "[#] Count should be equal to the number of samples in the scene."
    print("=" * 100)

    # Reverse the dict to have sample_token as key and scene_token as value
    scene2frames_token = {scene['token']: samples_token}
    frames2scene_token = {samples_token[i]: {'scene_token': scene['token'], 'timestamp': timestamps[i]} for i in range(len(samples_token))}
    
    return scene2frames_token, frames2scene_token
    # return {scene['token']: samples_token, 'timestamps': timestamps}


def get_truckscenes_dataset(cfg, deterministic=True):
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

    batch_size = 1 if cfg.training.single_sample_training else cfg.training.batch_size
    
    if deterministic:
        ts_dataloader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=batch_size,
            collate_fn=ts_dataset.collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    else:
        ts_dataloader = th.utils.data.DataLoader(
            ts_dataset,
            batch_size=batch_size,
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

        # Load all the scenes
        self.f2s_tokens = {}
        for scene in trucksc.scene:
            s2f_t, f2s_t = get_scene_token(trucksc, scene)
            self.f2s_tokens.update(f2s_t)

        self.f2s_tokens_list = list(self.f2s_tokens.keys())
        if cfg.training.single_sample_training:
            self.f2s_tokens_dict = dict(list(self.f2s_tokens.items())[:1])
            self.f2s_tokens_list = self.f2s_tokens_list[:1]

        self.radar_position = cfg.dataset.radar_position[0] # Assuming only one radar position is used
        self.padding_value = -1000

        # Configuration for image processing
        self.resize_ratio = cfg.condition_model.resize_ratio
        self.normalize_image = cfg.condition_model.normalize_image
        if not os.path.exists(f'{cfg.dataset.mean_sd_path}/mean_sd.json'):
            print(f"[#] {cfg.dataset.mean_sd_path}/mean_sd.json does not exist. Computing it...")
            self.pc_mean, self.pc_std = self.get_pc_mean_sd()
        else:
            print(f"[#] {cfg.dataset.mean_sd_path}/mean_sd.json exists. Loading it...")
            with open(f'{cfg.dataset.mean_sd_path}/mean_sd.json', 'r') as f:
                mean_std = json.load(f)
                self.pc_mean = np.array(mean_std['mean'])
                self.pc_std = np.array(mean_std['std'])

        print("[#] pc_mean: ", self.pc_mean)
        print("[#] pc_std: ", self.pc_std)

        self.__getitem__(0)
        
    def __len__(self):
        return len(self.f2s_tokens_list)
    
    def get_pc_mean_sd(self):
        """
        Calculate the mean and standard deviation of the point cloud data.
        
        Returns:
            tuple: Mean and standard deviation of the point cloud data.
        """
        all_pc = []

        for i in range(len(self.f2s_tokens_list)):
            frame_token = self.f2s_tokens_list[i]
            sample_record = self.trucksc.get('sample', frame_token)
            pointsensor_token = sample_record['data'][self.radar_position]
            pointsensor = self.trucksc.get('sample_data', pointsensor_token)
            pcl_path = os.path.join(self.trucksc.dataroot, pointsensor['filename'])
            # Load the point cloud
            pc = RadarPointCloud.from_file(pcl_path)    # 7xN; 7 is x, y, z, vx, vy, vz, rcs (radar cross section)
            pc_pts = pc.points.transpose(1, 0)  # Nx7
            pc_pts = pc_pts[:, :3]  # Only keep x, y, z
            all_pc.append(pc_pts)
        
        all_pc = np.concatenate(all_pc, axis=0)
        pc_mean = np.mean(all_pc, axis=0)[None, ...]    # 1x3
        pc_std = np.std(all_pc, axis=0)[None, ...]   # 1x3

        self.cfg.dataset.mean_sd_path = self.cfg.training.save_ckpt
        os.makedirs(self.cfg.dataset.mean_sd_path, exist_ok=True)
        with open(f'{self.cfg.dataset.mean_sd_path}/mean_sd.json', 'w') as f:
            print(f"[#] Saving mean and std to {self.cfg.dataset.mean_sd_path}/mean_sd.json")
            json.dump({'mean': pc_mean.tolist(), 'std': pc_std.tolist(), 'f2s_token_used':self.f2s_tokens_list, 'datetime':datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}, f)
        
        return pc_mean, pc_std

    def __getitem__(self, idx):
        frame_token = self.f2s_tokens_list[idx]
        sample_record = self.trucksc.get('sample', frame_token)
        pointsensor_token = sample_record['data'][self.radar_position]
        camera_token = sample_record['data'][SENSOR_PAIR_ID[self.radar_position]]
        cam = self.trucksc.get('sample_data', camera_token)
        pointsensor = self.trucksc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(self.trucksc.dataroot, pointsensor['filename'])
        
        # Load the point cloud
        pc = RadarPointCloud.from_file(pcl_path)    # 7xN; 7 is x, y, z, vx, vy, vz, rcs (radar cross section)
        pc_pts = pc.points.transpose(1, 0)  # Nx7
        pc_pts = pc_pts[:, :3]  # Only keep x, y, z
        # Load the camera image
        cam_img = Image.open(os.path.join(self.trucksc.dataroot, cam['filename']))  # HxWxC
        orig_w, orig_h = cam_img.size   # PIL’s .size is (width, height)
        new_w = int(orig_w  / self.resize_ratio)
        new_h = int(orig_h  / self.resize_ratio)
        cam_img = cam_img.resize((new_w, new_h), Image.LANCZOS)
        cam_img = np.array(cam_img)
        if self.normalize_image:
            # Normalize the image to [0, 1]
            cam_img = (cam_img / 255.0).astype(np.float32)
    
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

        # Z-normalization
        assert pc_pts.shape[1] == (3), f"[#] pc_pts shape: {pc_pts.shape} should be (N, 3)"
        assert self.pc_mean.shape == (1, 3), f"[#] pc_mean shape: {self.pc_mean.shape} should be (1, 3)"
        assert self.pc_std.shape == (1, 3), f"[#] pc_std shape: {self.pc_std.shape} should be (1, 3)"

        pc_pts = (pc_pts - self.pc_mean) / self.pc_std
        
        return pc_pts, cam_img, cam_dict

    def inv_transform(self, pc):
        """
        Inverse transform the point cloud data.
        
        Args:
            pc (ndarray): Point cloud data in shape (N, 3).
            
        Returns:
            ndarray: Inverse transformed point cloud data.
        """
        assert pc.shape[1] == (3), f"[#] pc shape: {pc.shape} should be (N, 3)"
        assert self.pc_mean.shape == (1, 3), f"[#] pc_mean shape: {self.pc_mean.shape} should be (1, 3)"
        assert self.pc_std.shape == (1, 3), f"[#] pc_std shape: {self.pc_std.shape} should be (1, 3)"

        pc = pc * self.pc_std + self.pc_mean
        return pc

    def collate_fn(self, batch):
        pc, img, cam_dict = map(list, zip(*batch))
        max_len = max([pc[i].shape[0] for i in range(len(pc))])
        if max_len > self.cfg.pointcloud_model.max_pc_len:
            max_len = self.cfg.pointcloud_model.max_pc_len
            
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
        for i in range(len(cam_dict)):
            for key in cam_dict[i].keys():
                R = cam_dict[i][key]['R'][None, None, ...]   # 1x1x3x3
                T = cam_dict[i][key]['T'][None, None, ...]   # 1x1x3
                R = np.repeat(R, repeats=max_len, axis=1)
                T = np.repeat(T, repeats=max_len, axis=1)
                out_cam_dict[key]['R'].append(R)
                out_cam_dict[key]['T'].append(T)
            
        for key in out_cam_dict.keys():
            out_cam_dict[key]['R'] = th.from_numpy(np.concatenate(out_cam_dict[key]['R'], axis=0))
            out_cam_dict[key]['T'] = th.from_numpy(np.concatenate(out_cam_dict[key]['T'], axis=0))
            # print(key, out_cam_dict[key]['R'].shape, out_cam_dict[key]['T'].shape)
    
        return {
            'pc': th.clone(pc_tensor),
            'img': th.clone(img_tensor),
            'mask': th.clone(mask_tensor),
            'cam_dict': out_cam_dict
        }