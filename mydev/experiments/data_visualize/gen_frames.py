import numpy as np
from PIL import Image
import os, tqdm
import matplotlib.pyplot as plt
from truckscenes import TruckScenes   
import argparse
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import sys
from utils import Visualizer
parser = argparse.ArgumentParser(description='Visualize TruckScenes dataset')
parser.add_argument('--version', type=str, default='v1.0-mini', help='dataset version')
parser.add_argument('--dataroot', type=str, default='/data/mint/Radar_Dataset/ManTruck/man-truckscenes/', help='dataset root path')
parser.add_argument('--out_path', type=str, default='./output/', help='output path')
args = parser.parse_args()

# Global variables
trucksc = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=True)
trucksc_visualizer = Visualizer(trucksc)

SENSOR_PAIR_ID = {
    # Left side
    'RADAR_LEFT_FRONT': 'CAMERA_LEFT_FRONT',
    'RADAR_LEFT_BACK': 'CAMERA_LEFT_BACK',
    # Right side
    'RADAR_RIGHT_FRONT': 'CAMERA_RIGHT_FRONT',
    'RADAR_RIGHT_BACK': 'CAMERA_RIGHT_BACK',

}



def get_all_frames(scene):
    print("=" * 100)
    # print("[#] scene:", scene)
    # print(scene.keys())
    # print("[#] First_sample_token: ", scene['first_sample_token'])
    # print("[#] Last_sample_token: ", scene['last_sample_token'])
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

    # return {scene['token']: samples_token, 'timestamps': timestamps}
    return {scene['token']: samples_token}


if __name__ == '__main__':
    print("[#] #N-scenes:", len(trucksc.scene))
    for sid in tqdm.tqdm(range(len(trucksc.scene)), desc="Processing scenes...", leave=False):
        samples_token = get_all_frames(trucksc.scene[sid])
        for k, v in samples_token.items():
            if k == 'timestamps':continue
            scene_id = k
            print("[#] #samples_token (frames):", len(v))
            print("[#] First sample token:", v[0])
            print("[#] Last sample token:", v[-1])
            print("=" * 100)
            os.makedirs(os.path.join(args.out_path, scene_id), exist_ok=True)
            for radar_channel, camera_channel in SENSOR_PAIR_ID.items():
                os.makedirs(os.path.join(args.out_path, scene_id, f'{camera_channel}_{radar_channel}'), exist_ok=True)
                for i in tqdm.tqdm(range(len(v)), desc=f"Processing {scene_id} on {camera_channel} and {radar_channel}", leave=False):
                    sample = trucksc.get('sample', v[i])
                    # if i == 0:
                    #     print("[#] Example of the first sample:")
                    #     print(sample.keys())
                    #     print(sample['data'].keys())
                        
                    # Loading radar data
                    radar_meta = trucksc.get('sample_data', sample['data'][radar_channel])
                    pcl_path = os.path.join(trucksc.dataroot, radar_meta['filename'])
                    radar_dat = RadarPointCloud.from_file(pcl_path)
                    radar_pc = radar_dat.points.transpose(1, 0)
                    
                    # Loading camera data
                    camera_meta = trucksc.get('sample_data', sample['data'][camera_channel])
                    camera_img = Image.open(os.path.join(trucksc.dataroot, camera_meta['filename']))
                    camera_img = np.array(camera_img)
                    
                    # Load the transformation matrix
                    sensor2ego_record = trucksc.get('calibrated_sensor', radar_meta['calibrated_sensor_token'])
                    cam2ego_record = trucksc.get('calibrated_sensor', camera_meta['calibrated_sensor_token'])
                    ego2world = trucksc.get('ego_pose', radar_meta['ego_pose_token'])
                    
                    points, coloring, img, fig, ax = trucksc_visualizer.render_pointcloud_in_image(
                        sample['token'], 
                        pointsensor_channel=radar_channel,
                        camera_channel=camera_channel,
                        render_intensity=True, 
                        dot_size=2)
                    
                    save_name = os.path.join(args.out_path, scene_id, f'{camera_channel}_{radar_channel}', f"{i:04d}.png")
                    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, transparent=True, dpi=800)
                    plt.close(fig)
                        # print("[#] points shape:", points.shape)
                        # print("[#] coloring shape:", coloring.shape)
                        # print("[#] img shape:", img.shape)
                        
                        # print("[#] Radar sensor2ego_record:", sensor2ego_record)
                        # print("[#] Camera sensor2ego_record:", cam2ego_record)
                        # print("[#] Ego2world:", ego2world)
                        # print("[#] Camera image shape:", camera_img.shape)
                        # print("[#] Radar data shape:", radar_pc.shape)
                        
                        # Performing the projection to image frame
                    # print("=" * 100)
            

