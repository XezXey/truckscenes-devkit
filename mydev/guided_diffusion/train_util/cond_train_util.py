import copy
import functools
import os, glob, plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import blobfile as bf
import torch as th
th.set_float32_matmul_precision('high')
import numpy as np
import tqdm
import torch.distributed as dist
from torchvision.utils import make_grid
from torch.optim import AdamW
# from pytorch_lightning import LightningModule
import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy

import lightning as pl
import wandb

from guided_diffusion import tensor_util#, vis_util, sampling_util

from .. import logger

from ..trainer_util import Trainer
from ..models.nn import update_ema
from ..resample import LossAwareSampler, UniformSampler
from ..script_util import seed_all, compare_models, dump_model_params
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_dict,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_dict = model_dict
        self.pointcloud_model = model_dict['pointcloud_model']
        if self.cfg.condition_model.apply:
            self.condition_model = model_dict['condition_model']

    def forward(self, trainloop, dat, cond):
        trainloop.run_step(dat, cond)


class TrainLoop(LightningModule):
    def __init__(
        self,
        *,
        model,
        name,
        diffusion,
        train_dataloader,
        train_dataset,
        cfg,
        t_logger,
        schedule_sampler=None,
    ):

        super(TrainLoop, self).__init__()
        self.cfg = cfg
        
        logger.configure(dir=cfg.logging.wandb_logger.dir)

        # Lightning
        self.n_gpus = self.cfg.training.n_gpus
        self.num_nodes = self.cfg.training.num_nodes
        self.t_logger = t_logger
        self.logger_mode = self.cfg.logging.logger
        
    
        self.pl_trainer = L.Trainer(
            devices=cfg.training.n_gpus,
            num_nodes=cfg.training.num_nodes,
            logger=t_logger,
            log_every_n_steps=cfg.logging.log_interval,
            max_epochs=int(cfg.training.max_epochs),
            accelerator=cfg.training.accelerator,
            profiler='simple',
            strategy=DDPStrategy(find_unused_parameters=cfg.training.find_unused_parameters),
            # detect_anomaly=True,
            )
        self.automatic_optimization = False # Manual optimization flow

        # Model
        assert len(model) == len(name)
        self.model_dict = {}
        for i, m in enumerate(model):
            self.model_dict[name[i]] = m

        self.model = ModelWrapper(model_dict=self.model_dict, cfg=self.cfg)

        # Diffusion
        self.diffusion = diffusion

        # Data
        self.ball_dataset = train_dataset
        self.train_loader = train_dataloader

        # Other config
        self.batch_size = self.cfg.training.batch_size
        self.lr = self.cfg.training.lr
        self.ema_rate = (
            [self.cfg.training.ema_rate]
            if isinstance(self.cfg.training.ema_rate, float)
            else [float(x) for x in self.cfg.training.ema_rate.split(",")]
        )
        self.log_interval = self.cfg.logging.log_interval
        self.save_interval = self.cfg.training.save_interval
        self.sampling_interval = self.cfg.training.sampling_interval
        self.resume_checkpoint = self.cfg.training.resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.global_batch = self.n_gpus * self.batch_size * self.num_nodes
        self.weight_decay = self.cfg.training.weight_decay
        self.lr_anneal_steps = self.cfg.training.lr_anneal_steps
        self.name = name

        self.step = 0
        self.resume_step = 0
        
        # Load model checkpoints
        self.load_ckpt()

        self.model_trainer_dict = {}
        for name, model in self.model_dict.items():
            self.model_trainer_dict[name] = Trainer(name=name, model=model, pl_module=self)

        self.opt = AdamW(
            sum([list(self.model_trainer_dict[name].master_params) for name in self.model_trainer_dict.keys()], []),
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Initialize ema_parameters
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.model_ema_params_dict = {}
            for name in self.model_trainer_dict.keys():
                self.model_ema_params_dict[name] = [
                    self._load_ema_parameters(rate=rate, name=name) for rate in self.ema_rate
                ]

        else:
            self.model_ema_params_dict = {}
            for name in self.model_trainer_dict.keys():
                self.model_ema_params_dict[name] = [
                    copy.deepcopy(self.model_trainer_dict[name].master_params) for _ in range(len(self.ema_rate))
                ]
    
    def load_ckpt(self):
        '''
        Load model checkpoint from filename = model{step}.pt
        '''
        found_resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, k="model", model_name=self.model_dict.keys())
        if found_resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            for name in self.model_dict.keys():
                ckpt_path = found_resume_checkpoint[f'{name}_model']
                # logger.log(f"Loading model checkpoint (name={name}, step={self.resume_step}): {ckpt_path}")
                self.model_dict[name].load_state_dict(
                    th.load(ckpt_path, map_location='cpu'),
                )
        elif (self.resume_checkpoint != "") and (not found_resume_checkpoint):
            assert FileNotFoundError(f"[#] Checkpoint not found on {self.resume_checkpoint}")

    def _load_optimizer_state(self):
        '''
        Load optimizer state
        '''
        found_resume_opt = find_resume_checkpoint(self.resume_checkpoint, k="opt", model_name=['opt'])
        if found_resume_opt:
            opt_path =found_resume_opt['opt_opt']
            print(f"Loading optimizer state from checkpoint: {opt_path}")
            self.opt.load_state_dict(
                th.load(opt_path, map_location='cpu'),
            )
    
    def _load_ema_parameters(self, rate, name):

        found_resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, k=f"ema_{rate}", model_name=[name])
        if found_resume_checkpoint:
            ckpt_path = found_resume_checkpoint[f'{name}_ema_{rate}']
            print(f"Loading EMA from checkpoint: {ckpt_path}...")
            state_dict = th.load(ckpt_path, map_location='cpu')
            ema_params = self.model_trainer_dict[name].state_dict_to_master_params(state_dict)

        return ema_params

    def run(self):
        # Driven code
        # Logging for first time
        if not self.resume_checkpoint:
            self.save()
        self.pl_trainer.fit(self, train_dataloaders=self.train_loader)


    def run_step(self, dat, cond):
        '''
        1-Training step
        :params dat: the image data in BxCxHxW
        :params cond: the condition dict e.g. ['cond_params'] in BXD; D is dimension of DECA, Latent, ArcFace, etc.
        '''
        self.zero_grad_trainer()
        self.forward_backward(dat, cond)
        took_step = self.optimize_trainer()
        self.took_step = took_step


    def training_step(self, batch, batch_idx):
        # dat = batch['traj_3d']
        pc = batch['pc']
        img = batch['img']
        self.model(trainloop=self, dat=pc, cond=img)
        self.step += 1
    
    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx):
        '''
        callbacks every training step ends
        1. update ema (Update after the optimizer.step())
        2. logs
        '''
        if self.took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_rank_zero(batch)
        self.save_rank_zero()

        # Reset took_step flag
        self.took_step = False
    
    @rank_zero_only 
    def save_rank_zero(self):
        if self.step % self.save_interval == 0:
            self.save()

    @rank_zero_only
    def log_rank_zero(self, batch):
        if self.step % self.log_interval == 0:
            self.log_step()
        if (self.step % self.sampling_interval == 0) or (self.resume_step!=0 and self.step==1) :
            self.log_sampling(batch, sampling_model='ema')
            self.log_sampling(batch, sampling_model='model')
    
    def zero_grad_trainer(self):
        for name in self.model_trainer_dict.keys():
            self.model_trainer_dict[name].zero_grad()
        self.opt.zero_grad()


    def optimize_trainer(self):
        self.opt.step()
        for name in self.model_trainer_dict.keys():
            self.model_trainer_dict[name].get_norms()
        return True

    def forward_cond_network(self, cond, model_dict=None):
        #NOTE: Dev this, if we really the condition model
        if model_dict is None:
            model_dict = self.model_dict
            
        if self.cfg.condition_model.apply:
            out_cond = model_dict[self.cfg.condition_model.name](
                x=cond.float(), 
                emb=None,
            )
            # Override the condition and re-create cond_params
            if self.cfg.condition_model.override_cond != "":
                cond[self.cfg.condition_model.override_cond] = out_cond
                tmp = []
                for p in self.cfg.trajectory_model.cond_selector:
                    tmp.append(cond[p])
                cond[''] = th.cat(tmp, dim=-1)
            else: raise NotImplementedError
        return cond

    def prepare_cond(self, cond):
        if self.cfg.trajectory_model.dpm_conditioning:
            cond['dpm_condition'] = cond['traj_condition']
        else: cond['dpm_condition'] = None
        return cond
    
    def forward_backward(self, dat, cond):

        t, weights = self.schedule_sampler.sample(dat.shape[0], self.device)
        # Expand dims for broadcasting sicne we also have timestep dims e.g. (B, T, #features)
        noise = th.randn_like(dat)
        # print(noise.shape, noise, dat.shape, dat)
        
        cond = self.forward_cond_network(cond)
        cond = self.prepare_cond(cond)
        
        # Losses
        model_compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model_dict[self.cfg.trajectory_model.name],
            dat,
            t,
            noise=noise,
            model_kwargs=cond,
            cfg=self.cfg,
            dataset=self.ball_dataset
        )
        model_losses, _ = model_compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, model_losses["loss"].detach()
            )

        # Diffusion loss
        loss = (model_losses["loss"] * weights).mean()
        self.manual_backward(loss)
        if self.step % self.log_interval:
            self.log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in model_losses.items()}, module=self.cfg.trajectory_model.name,
            )

    @rank_zero_only
    def _update_ema(self):
        for name in self.model_ema_params_dict:
            for rate, params in zip(self.ema_rate, self.model_ema_params_dict[name]):
                    update_ema(params, self.model_trainer_dict[name].master_params, rate=rate)

    def _anneal_lr(self):
        '''
        Default set to 0 => No lr_anneal step
        '''
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    @rank_zero_only
    def log_step(self):
        step_ = float(self.step + self.resume_step)
        self.log("training_progress/step", step_ + 1)
        self.log("training_progress/global_step", (step_ + 1) * self.n_gpus * self.num_nodes)
        self.log("training_progress/global_samples", (step_ + 1) * self.global_batch)

    @rank_zero_only
    def eval_mode(self, model):
        for _, v in model.items():
            v.eval()

    @rank_zero_only
    def train_mode(self, model):
        for _, v in model.items():
            v.train()
        

    @rank_zero_only
    def log_sampling(self, batch, sampling_model):
        def get_ema_model(rate):
            rate_idx = self.ema_rate.index(rate)
            ema_model = copy.deepcopy(self.model_dict)
            ema_params = dict.fromkeys(self.model_ema_params_dict.keys())
            for name in ema_params:
                ema_params[name] = self.model_trainer_dict[name].master_params_to_state_dict(self.model_ema_params_dict[name][rate_idx])
                ema_model[name].load_state_dict(ema_params[name])
            
            return ema_model
        
        print(f"Sampling with {sampling_model}...")
        
        if sampling_model == 'ema':
            ema_model_dict = get_ema_model(rate=0.9999)
            sampling_model_dict = ema_model_dict
        elif sampling_model == 'model':
            sampling_model_dict = self.model_dict
        else: raise NotImplementedError("Only \"model\" or \"ema\"")
        
        self.eval_mode(model=sampling_model_dict)

        step_ = float(self.step + self.resume_step)

        n = self.cfg.train.n_sampling
        if self.cfg.train.same_sampling:
            # batch here is a tuple of (dat, cond); thus used batch[0], batch[1] here
            testing_trajectory = [self.ball_dataset.__getsingle__(i) for i in range(n)]
        else:
            testing_trajectory = [self.ball_dataset.__getsingle__(i) for i in np.random.randint(0, self.ball_dataset.__len__(), n)]
            
        prediction = []
        gt = []
        for i in tqdm.tqdm(range(len(testing_trajectory))):
            # Full trajectory sampling
            trajectory = th.tensor(testing_trajectory[i]['traj_3d_raw'][None, ...]).cuda()
            trajectory_model_input = th.tensor(testing_trajectory[i]['traj_model_input'][None, ...]).cuda()
            reconstruction_info = testing_trajectory[i]['reconstruction_info']
            cond = {'traj_condition' : th.tensor(testing_trajectory[i]['traj_condition'][None, ...]).cuda()}
            # Some preprocessing on conditions
            cond = self.forward_cond_network(cond)
            cond = self.prepare_cond(cond)
            # 1. Overlapped Sampling
            pred_overlap, _ = sampling_util.overlap_sampling(trajectory=trajectory_model_input,
                                                        cond=cond, 
                                                        model=sampling_model_dict[self.cfg.trajectory_model.name], 
                                                        cfg=self.cfg,
                                                        sample_fn=self.diffusion.p_sample_loop,
                                                        # sample_fn=self.diffusion.ddim_sample_loop,
                                                        device=self.device,
                                                        dataset=self.ball_dataset,
                                                    )
            pred_overlap = pred_overlap[None, ...]
            if self.cfg.trajectory_model.out_prediction == ['y']:
                output = utils_transform.reconstruct(height=pred_overlap, 
                                                     recon_dict=reconstruction_info, 
                                                     use_canonicalize=self.cfg.trajectory_model.use_canonicalize,)
            elif self.cfg.trajectory_model.out_prediction == ['x', 'y', 'z']:
                output = pred_overlap
            else: raise NotImplementedError("Only \"y\" or \"x, y, z\"")
            assert trajectory.shape == output.shape
            output = output.cpu().float()
            prediction.append(output)
            gt.append(trajectory.type_as(output))
        
        # Logging Trajectory
        # fig = vis_util.plot_border()
        fig3d = go.Figure()
        fig2d_axis = make_subplots(rows=3, cols=1)
        for i in range(n):
            # Ground Truth & Predicted Trajectory on 3D
            fig3d = vis_util.plot_3d(gt[i], c='rgb(0, 0, 255)', fig=fig3d, prefix=f'gt-{i}')
            fig3d = vis_util.plot_3d(prediction[i], c='rgb(255, 0, 0)', fig=fig3d, prefix=f'pred-{i}')
            fig3d = vis_util.plot_ray(dat=testing_trajectory[i]['reconstruction_info'], id=i, fig=fig3d)
            # Ground Truth & Predicted Trajectory on 2D
            fig2d_axis = vis_util.plot_2d(gt[i], c='rgb(0, 0, 255)', fig=fig2d_axis, prefix=f'gt-{i}')
            fig2d_axis = vis_util.plot_2d(prediction[i], c='rgb(255, 0, 0)', fig=fig2d_axis, prefix=f'pred-{i}')
            
        
        self.trainer.logger.experiment.log({
            f"Trajectory 3D - {sampling_model}": wandb.Html(plotly.io.to_html(fig3d))
        }, step = int((step_ + 1) * self.n_gpus))
        
        self.trainer.logger.experiment.log({
            f"Trajectory 2D - {sampling_model}": wandb.Html(plotly.io.to_html(fig2d_axis))
        }, step = int((step_ + 1) * self.n_gpus))
        
        # Compute MSE
        x_error, y_error, z_error, trajectory_error = compute_mse(gt=th.cat(gt, dim=1), pred=th.cat(prediction, dim=1))
        self.log(f'MSELoss/{sampling_model}/Trajectory', trajectory_error)
        self.log(f'MSELoss/{sampling_model}/X', x_error)
        self.log(f'MSELoss/{sampling_model}/Y', y_error)
        self.log(f'MSELoss/{sampling_model}/Z', z_error)
        
                                                                              
        # # Save memory!
        self.train_mode(model=sampling_model_dict)

    @rank_zero_only
    def save(self):
        save_step = self.step + self.resume_step
        def save_checkpoint(rate, params, trainer, name=""):
            state_dict = trainer.master_params_to_state_dict(params)
            if not rate:
                filename = f"{name}_model{save_step:06d}.pt"
            else:
                filename = f"{name}_ema_{rate}_{save_step:06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for name in self.model_dict.keys():
            save_checkpoint(0, self.model_trainer_dict[name].master_params, self.model_trainer_dict[name], name=name)
            for rate, params in zip(self.ema_rate, self.model_ema_params_dict[name]):
                save_checkpoint(rate, params, self.model_trainer_dict[name], name=name)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{save_step:06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def configure_optimizers(self):
        print("[#] Optimizer")
        print(self.opt)
        return self.opt

    @rank_zero_only
    def log_loss_dict(self, diffusion, ts, losses, module):
        for key, values in losses.items():
            self.log(f"training_loss_{module}/{key}", values.mean().item())
            if key == "loss":
                self.log(f"{key}", values.mean().item(), prog_bar=True, logger=False)
            # log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"training_loss_{module}/{key}_q{quartile}", sub_loss)
                if key == "loss":
                    self.log(f"{key}_q{quartile}", sub_loss, prog_bar=True, logger=False)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def find_resume_checkpoint(ckpt_dir, k, model_name):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    """
    Find resume checkpoint from {ckpt_dir} and search for {k}
    :param ckpt_dir: checkpoint directory (Need to input with the model{...}.pt)
    :param k: keyword to find the checkpoint e.g. 'model', 'ema', ...
    :param step: step of checkpoint (this retrieve from the ckpt_dir)
    """
    step = parse_resume_step_from_filename(ckpt_dir)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_dir))
    all_ckpt = glob.glob(f"{ckpt_dir}/*{step}.pt")  # List all checkpoint give step.
    found_ckpt = {}
    for name in model_name:
        for c in all_ckpt:
            if (k in c.split('/')[-1]) and (name in c.split('/')[-1]):
                found_ckpt[f"{name}_{k}"] = c
                assert bf.exists(found_ckpt[f"{name}_{k}"])
    return found_ckpt

def find_ema_checkpoint(main_checkpoint, step, rate, name):
    if main_checkpoint is None:
        return None
    filename = f"{name}_ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None