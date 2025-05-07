import torch as th
# import pytorch_lightning as pl
import lightning as L
import sys, os, tqdm
sys.path.append("/home/mint/Dev/RadarPointcloud/truckscenes-devkit/mydev/")
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
# from pytorch_lightning.strategies import DDPStrategy
# from L.strategies import DDPStrategy




seed_all(25091995)

if __name__ == "__main__":
    
    # Init config
    cfg = parse_args()
    if cfg.training.save_name is None:
        print("Please specify the save_name in the config file...")
        exit()
    cfg.training.save_ckpt = os.path.join(cfg.training.save_ckpt, cfg.training.save_name)
    cfg.training.visualization = os.path.join(cfg.training.visualization, cfg.training.save_name)
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
    
    # train_dataloader, train_dataset = get_truckscenes_dataset(
    #     cfg=cfg,
    # )
    
    train_dataloader, train_dataset = get_random_dataset()
    
    # for i, data in enumerate(train_dataloader):
    #     print("MINT")
    
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
    
    pl_trainer = L.Trainer(
        devices=cfg.training.n_gpus,
        num_nodes=cfg.training.num_nodes,
        logger=wandb_logger,
        log_every_n_steps=cfg.logging.log_interval,
        max_epochs=int(cfg.training.max_epochs),
        accelerator=cfg.training.accelerator,
        profiler='simple',
        # strategy=DDPStrategy(find_unused_parameters=cfg.training.find_unused_parameters),
        strategy='ddp',
        # detect_anomaly=True,
        )
    
    # pl_trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    # pl_trainer.fit(model=get_dummy_model(), train_dataloaders=train_dataloader)
    pl_trainer.fit(model=train_loop, train_dataloaders=train_dataloader)
    
    