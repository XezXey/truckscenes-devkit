'''
Default config for Where is the Ball? codebase
'''
from re import A
from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import datetime

cfg = CN()

cfg.name = "Radar PointCloud Reconstruction"
cfg.device = 'cuda'
cfg.device_id = '0'

# ---------------------------------------------------------------------------- #
# Options for network architecture
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.name = 'default_model'

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.training = CN()
cfg.training.max_pc_len = 128
cfg.training.batch_size = 8
cfg.training.max_epochs = int(1e17)
cfg.training.accelerator = 'gpu'
cfg.training.lr = 0.001
cfg.training.save_ckpt = '../model_checkpoints/'
cfg.training.load_ckpt = '../model_checkpoints/'
cfg.training.visualization = '../training_visualization/'
cfg.training.save_name = None
cfg.training.save_interval = 5000 # Save model every n iterations
cfg.training.save_ema_interval = 1000 # Save model every n iterations
cfg.training.eval_interval = 50 # Evaluate model on validation set every n iterations
cfg.training.ema_rate = 0.9999

# ---------------------------------------------------------------------------- #
# Options for loss
# ---------------------------------------------------------------------------- #
cfg.logging = CN()
cfg.logging.log_interval = 50
cfg.logging.visualize_log_interval = 1000000
cfg.logging.logger = 'wandb'
cfg.logging.wandb_logger = CN()
cfg.logging.wandb_logger.notes = 'Default note...'
cfg.logging.wandb_logger.project_name = 'Radar Pointcloud Reconstruction'
cfg.logging.wandb_logger.tags = ['default']
cfg.logging.wandb_logger.run_name = 'test'
cfg.logging.wandb_logger.dir = './'
cfg.logging.wandb_logger.run_mode = 'run'
cfg.logging.wandb_logger.resume = None

# ---------------------------------------------------------------------------- #
# Options for loss
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.loss_list = ['all']

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.version = 'v1.0-mini'
cfg.dataset.dataroot = '/data/mint/Radar_Dataset/ManTruck/man-truckscenes/'
cfg.dataset.sample_token = "first_sample_token"
cfg.dataset.radar_position = ['RADAR_LEFT_FRONT']

# ---------------------------------------------------------------------------- #
# Options for Evaluation
# ---------------------------------------------------------------------------- #
cfg.eval = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone dict so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()  # Dict of default configs

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def cmd_to_cfg_format(opts):
    """
    Override config from a list
    src-format : ['--dataset.train', '/data/mint/dataset']
    dst-format : ['dataset.train', '/data/mint/dataset']
    for writing a "dataset.train" key
    """
    opts_new = []
    for i, opt in enumerate(opts):
        if (i+1) % 2 != 0:
            opts_new.append(opt[2:])
        else: 
            opts_new.append(opt)
    return opts_new

def parse_args(ipynb={'mode':False, 'cfg':None}):
    '''
    Return dict-like cfg, accesible with cfg.<key1>.<key2> or cfg[<key1>][<key2>]
    e.g. <key1> = dataset, <key2> = training_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args, opts = parser.parse_known_args()
    if ipynb['mode']:
        # Using this with ipynb will have some opts defaults from ipynb and we need to filter out.
        opts=[]
        args.cfg = ipynb['cfg']

    print("Merging with : ", args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    # Merge with cmd-line argument(s)
    if opts != []:
        cfg_list = cmd_to_cfg_format(opts)
        cfg.merge_from_list(cfg_list)

    # # Some parameters in config need to be updated
    # cfg = update_params(cfg)
    # # Update the dataset path
    # cfg = update_dataset_path(cfg)
    return cfg

def parse_args_testing(ipynb={'mode':False, 'cfg':None}):
    '''
    Return dict-like cfg, accesible with cfg.<key1>.<key2> or cfg[<key1>][<key2>]
    e.g. <key1> = dataset, <key2> = training_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    # parser.add_argument('--ckpt', type=str, help='ckpt directory')
    args, opts = parser.parse_known_args()
    if ipynb['mode']:
        # Using this with ipynb will have some opts defaults from ipynb and we need to filter out.
        opts=[]
        args.cfg = ipynb['cfg']

    print("Merging with : ", args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    # Merge with cmd-line argument(s)
    if opts != []:
        cfg_list = cmd_to_cfg_format(opts)
        cfg.merge_from_list(cfg_list)

    # # Some parameters in config need to be updated
    # cfg = update_params(cfg)
    # # Update the dataset path
    # cfg = update_dataset_path(cfg)
    return cfg

if __name__ == '__main__':
    print(get_cfg_defaults())