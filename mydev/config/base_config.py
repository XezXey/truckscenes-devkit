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
# For MDM
cfg.pointcloud_model = CN()
cfg.pointcloud_model.name = 'pointcloud_model'
cfg.pointcloud_model.arch = 'MDM'
cfg.pointcloud_model.num_layers = 6
cfg.pointcloud_model.num_heads = 8
cfg.pointcloud_model.ff_size = 2048
cfg.pointcloud_model.model_channels = 512
cfg.pointcloud_model.in_channels = 3
cfg.pointcloud_model.out_channels = 3
cfg.pointcloud_model.dropout = 0.1
cfg.pointcloud_model.max_pc_len = 128
cfg.pointcloud_model.norm_first = False
cfg.pointcloud_model.predict_velocity = True

# Conditioning model (e.g. encoder, feature extractor, etc.)
cfg.condition_model = CN()
cfg.condition_model.name = 'condition_model'
cfg.condition_model.arch = 'EncoderUNetModelNoTime'
cfg.condition_model.apply = False 
cfg.condition_model.resize_ratio = 1.0
cfg.condition_model.normalize_image = True
cfg.condition_model.image_size = 512
cfg.condition_model.in_channels = 3
cfg.condition_model.model_channels = 128
cfg.condition_model.out_channels = 128
cfg.condition_model.num_res_blocks = 2
cfg.condition_model.num_heads = 4
cfg.condition_model.num_heads_upsample = -1
cfg.condition_model.num_head_channels = -1
cfg.condition_model.attention_resolutions = "16,8"
cfg.condition_model.channel_mult = ""
cfg.condition_model.dropout = 0.0
cfg.condition_model.use_checkpoint = False
cfg.condition_model.use_scale_shift_norm = True
cfg.condition_model.resblock_updown = False
cfg.condition_model.use_new_attention_order = False
cfg.condition_model.pool = 'adaptive'

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.training = CN()
cfg.training.batch_size = 8
cfg.training.n_gpus = 1
cfg.training.num_nodes = 1
cfg.training.max_epochs = int(1e17)
cfg.training.accelerator = 'gpu'
cfg.training.lr = 1e-4
cfg.training.lr_anneal_steps = 0.0
cfg.training.weight_decay = 0.0
cfg.training.save_ckpt = '/data/mint/RadarPC/model_logs/'
cfg.training.load_ckpt = '../model_checkpoints/'
cfg.training.visualization = '/data/mint/RadarPC/visualize_logs/'
cfg.training.n_sampling = 1
cfg.training.same_sampling = True
cfg.training.save_name = None
cfg.training.save_interval = 5000 # Save model every n iterations
cfg.training.save_ema_interval = 1000 # Save model every n iterations
cfg.training.sampling_interval = 1000 # 
cfg.training.single_sample_training = False
cfg.training.single_sample_training_expand = 256
cfg.training.eval_interval = 50 # Evaluate model on validation set every n iterations
cfg.training.ema_rate = 0.9999
cfg.training.resume_checkpoint = ""
cfg.training.find_unused_parameters = True

# ---------------------------------------------------------------------------- #
# Options for loss
# ---------------------------------------------------------------------------- #
cfg.logging = CN()
cfg.logging.log_interval = 2
cfg.logging.visualize_log_interval = 1000000
cfg.logging.logger = 'wandb'
cfg.logging.wandb_logger = CN()
cfg.logging.wandb_logger.notes = 'Default note...'
cfg.logging.wandb_logger.project_name = 'Radar Pointcloud Reconstruction'
cfg.logging.wandb_logger.tags = ['default']
cfg.logging.wandb_logger.run_name = 'test'
cfg.logging.wandb_logger.dir = '/data/mint/RadarPC/wandb_logs/'
cfg.logging.wandb_logger.run_mode = 'run'
cfg.logging.wandb_logger.resume = None

# ---------------------------------------------------------------------------- #
# Options for loss
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.loss_list = ['all']

# ---------------------------------------------------------------------------- #
# Options for diffusion
# ---------------------------------------------------------------------------- #
cfg.diffusion = CN()
cfg.diffusion.schedule_sampler = "uniform"
cfg.diffusion.learn_sigma = False
cfg.diffusion.diffusion_steps = 1000
cfg.diffusion.sigma_small = False
cfg.diffusion.noise_schedule = "linear"
cfg.diffusion.use_kl = False
cfg.diffusion.predict_xstart = False
cfg.diffusion.rescale_timesteps = False
cfg.diffusion.rescale_learned_sigmas = False
cfg.diffusion.timestep_respacing = ""
cfg.diffusion.clip_denoised = True


# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.version = 'v1.0-mini'
cfg.dataset.dataroot = '/data/mint/Radar_Dataset/ManTruck/man-truckscenes/'
cfg.dataset.sample_token = "first_sample_token"
cfg.dataset.radar_position = ['RADAR_LEFT_FRONT']
cfg.dataset.mean_sd_path = None
cfg.dataset.mean = ''
cfg.dataset.std = ''

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