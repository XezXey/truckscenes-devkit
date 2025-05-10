import torch as th
# import pytorch_lightning as pl
import lightning as L
import sys, os, tqdm
sys.path.append("/home/mint/Dev/Radar_Pointcloud_Recon/truckscenes-devkit/mydev/")
from guided_diffusion.dataloader.dataloader import get_truckscenes_dataset
from guided_diffusion.dataloader.random_dataloader import get_random_dataset
from config.base_config import parse_args
from lightning.pytorch.loggers import WandbLogger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    seed_all,
)
from guided_diffusion.train_util.cond_train_util import TrainLoop
from guided_diffusion.train_util.train_dummy_util import get_dummy_model




seed_all(25091995)

if __name__ == "__main__":
    
    # Init config
    cfg = parse_args()
    if cfg.training.save_name is None:
        print("Please specify the save_name in the config file...")
        exit()
    cfg.training.save_ckpt = os.path.join(cfg.training.save_ckpt, cfg.training.save_name, 'ckpt')
    cfg.training.visualization = os.path.join(cfg.training.visualization, cfg.training.save_name, 'visualization')
    cfg.logging.wandb_logger.dir = os.path.join(cfg.logging.wandb_logger.dir, cfg.training.save_name)
    
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
    
    
    train_loop = TrainLoop(
        model=list(model.values()),
        name=list(model.keys()),
        diffusion=diffusion,
        train_dataloader=train_dataloader,
        train_dataset=train_dataset,
        cfg=cfg,
        t_logger=wandb_logger,
        schedule_sampler=schedule_sampler,
    )
    
    train_loop.run()
    
    
