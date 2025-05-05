import torch as th
import pytorch_lightning as pl
import sys
sys.path.append("/home/mint/Dev/RadarPointcloud/truckscenes-devkit/mydev/")
from dataloader.dataloader import get_truckscenes_dataset

if __name__ == "__main__":
    get_truckscenes_dataset(
        root_dir="/data/mint/Radar_Dataset/ManTruck/man-truckscenes/",
        version="v1.0-mini",
    )
    