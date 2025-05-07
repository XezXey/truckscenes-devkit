import torch as th
import pytorch_lightning as pl
import sys, os, tqdm
sys.path.append("/home/mint/Dev/RadarPointcloud/truckscenes-devkit/mydev/")
from dataloader.dataloader import get_truckscenes_dataset
from config.base_config import parse_args
from pytorch_lightning.loggers import WandbLogger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    seed_all,
)


seed_all(25091995)

if __name__ == "__main__":
    
    # Init config
    cfg = parse_args()
    if cfg.training.save_name is None:
        print("Please specify the save_name in the config file...")
        exit()
    cfg.training.save_ckpt = os.path.join(cfg.training.save_ckpt, cfg.training.save_name)
    cfg.training.visualization = os.path.join(cfg.training.visualization, cfg.training.save_name)
    
    # Init diffusion, model and schedule sampler
    print("[#] Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(cfg)
    # Filter the "None" model out
    model = {k: v for k, v in model.items() if v is not None}
    print("[#] Model", model)
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    print("[#] Creating data loader...")
    
    # Init wandb
    logger = cfg.logging.wandb_logger

    os.environ['WANDB_MODE'] = logger.run_mode
    wandb_logger = WandbLogger(project=logger.project_name, save_dir=logger.dir, tags=logger.tags, name=logger.run_name, notes=logger.notes)
    
    train_dataloader, train_dataset = get_truckscenes_dataset(
        cfg=cfg,
    )
    for i, data in enumerate(train_dataloader):
        
        print("MINT")
    
    