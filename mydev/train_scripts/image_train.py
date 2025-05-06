import torch as th
import pytorch_lightning as pl
import sys, os, tqdm
sys.path.append("/home/mint/Dev/RadarPointcloud/truckscenes-devkit/mydev/")
from dataloader.dataloader import get_truckscenes_dataset
from config.base_config import parse_args
from pytorch_lightning.loggers import WandbLogger


pl.seed_everything(25091995, workers=True)

if __name__ == "__main__":
    
    cfg = parse_args()
    if cfg.training.save_name is None:
        print("Please specify the save_name in the config file...")
        exit()
    cfg.training.save_ckpt = os.path.join(cfg.training.save_ckpt, cfg.training.save_name)
    cfg.training.visualization = os.path.join(cfg.training.visualization, cfg.training.save_name)
    
    # Init wandb
    logger = cfg.logging.wandb_logger

    os.environ['WANDB_MODE'] = logger.run_mode
    wandb_logger = WandbLogger(project=logger.project_name, save_dir=logger.dir, tags=logger.tags, name=logger.run_name, notes=logger.notes)
    
    train_dataloader, train_dataset = get_truckscenes_dataset(
        cfg=cfg,
    )
    for i, data in enumerate(train_dataloader):
        
        print("MINT")
    
    