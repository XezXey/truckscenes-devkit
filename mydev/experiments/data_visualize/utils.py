from matplotlib.colors import Colormap, Normalize
import os
import os.path as osp

from datetime import datetime
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from PIL import Image
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from pyquaternion import Quaternion

from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import view_points, transform_matrix, \
    BoxVisibility


class Visualizer:
    def __init__(self, trucksc):
        self.trucksc = trucksc
        
    def map_pointcloud_to_image(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                cmap: str = 'viridis',
                                cnorm: bool = True) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token,
        load pointcloud and map it to the image plane.

        Arguments:
            pointsensor_token: Lidar/radar sample_data token.
            camera_token: Camera sample_data token.
            min_dist: Distance from the camera below which points are discarded.
            render_intensity: Whether to render lidar intensity instead of point depth.

        Returns:
            (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        if not isinstance(cmap, Colormap):
            cmap = plt.get_c[cmap]

        cam = self.trucksc.get('sample_data', camera_token)
        pointsensor = self.trucksc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.trucksc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.trucksc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed
        # via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = self.trucksc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.trucksc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame
        # for the timestamp of the image.
        poserecord = self.trucksc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            if pointsensor['sensor_modality'] == 'lidar':
                # Retrieve the color from the intensities.
                coloring = pc.points[3, :]
            else:
                # Retrieve the color from the rcs.
                coloring = pc.points[6, :]
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Color mapping
        if cnorm:
            norm = Normalize(vmin=np.quantile(coloring, 0.5),
                             vmax=np.quantile(coloring, 0.95), clip=True)
        else:
            norm = None
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        coloring = mapper.to_rgba(coloring)[..., :3]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']),
                             normalize=True)

        # Remove points that are either outside or behind the camera.
        # Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to
        # avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask, :]

        return points, coloring, im

    def render_pointcloud_in_image(self,
                                sample_token: str,
                                dot_size: int = 4,
                                pointsensor_channel: str = 'LIDAR_LEFT',
                                camera_channel: str = 'CAMERA_LEFT_FRONT',
                                render_intensity: bool = False,
                                ax: Axes = None,
                                cmap: str = 'viridis',
                            ):
            """
            Scatter-plots a pointcloud on top of image.

            Arguments:
                sample_token: Sample token.
                dot_size: Scatter plot dot size.
                pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_LEFT'.
                camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
                out_path: Optional path to save the rendered figure to disk.
                render_intensity: Whether to render lidar intensity instead of point depth.
                ax: Axes onto which to render.
                verbose: Whether to display the image in a window.
            """
            if not isinstance(cmap, Colormap):
                cmap = cm.get_cmap(cmap)

            sample_record = self.trucksc.get('sample', sample_token)

            # Here we just grab the front camera and the point sensor.
            pointsensor_token = sample_record['data'][pointsensor_channel]
            camera_token = sample_record['data'][camera_channel]

            points, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                                render_intensity=render_intensity,
                                                                cmap=cmap)
            
            # Init axes.
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(9, 16))
                ax.set_title(sample_token)
            else:  # Set title on if rendering as part of render_sample.
                ax.set_title(camera_channel)
            ax.imshow(im)
            ax.scatter(points[0, :], points[1, :], marker='o', c=coloring,
                       s=dot_size, edgecolors='none')
            ax.axis('off')
            plt.tight_layout()
            # plt.savefig('render_img.png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=800)
            
            

            return points, coloring, im, fig, ax