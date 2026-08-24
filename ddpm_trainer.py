"""
This module contains training routines for the DDPM/DDIM U-Net diffusion model.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import torch
from torch import Tensor
import psutil
import math, copy
import torch.nn as nn
from functools import wraps
from tqdm.auto import tqdm
from typing import Tuple, Callable, Dict, List
import logging, gc
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import get_device, get_amp_dtype, generate_loss_plots, save_images
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from dataset_utils import get_class_labels


def infinite_loader(dataloader: DataLoader):
    """
    Infinitely yields batches of data from the input dataloader (dl) without caching batches.
    """
    while True:
        for batch in dataloader:
            yield batch


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def compute_with_amp(func):
    """
    Decorator that wraps a func in automatic mixed precision (AMP) evaluation context managers from
    pytorch if self.amp_dtype is not None. This does not alter the inputs or outputs of func.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.amp_dtype is not None:
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Sets the requires_grad_ property for all parameters in a given module.

    :param module: The input module whose parameters will be affected.
    :param requires_grad: True or False indicating if gradient tracking should be enabled or disabled.
    :returns: None, the internal state of module is edited inplace.
    """
    for p in module.parameters():
        p.requires_grad_(requires_grad)


class Trainer:

    def __init__(self, config: Dict, dataloaders: Dict[str, DataLoader], **kwargs):
        """
        Creates a Trainer object for training the U-Net model using the DDPM objective. This class has methods
        for training, saving, and loading model checkpoints.

        :param config: An input config dictionary file detailing the configuration parameters.
        """
        super().__init__()
        self.config = config  # Record config parameters passed
        self.num_classes = config["UNet"]["num_classes"]
        self.class_labels = get_class_labels(config["dataset"])  # A dict mapping int:str for each class label

        ### Set up directories for the output (sampled images), losses, and model checkpoints
        self.results_dir = os.path.join(CURRENT_DIR, "results", str(config["name"]))
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.losses_dir = os.path.join(self.results_dir, "losses")
        self.samples_dir = os.path.join(self.results_dir, "samples")
        for directory in [self.results_dir, self.checkpoints_dir, self.losses_dir, self.samples_dir]:
            os.makedirs(directory, exist_ok=True)  # Create the directories needed if not already there

        #### Set up logging during training
        self.configure_logging()
        self.logger.info("Initializing DDPM Trainer")
        timesteps = config["GaussianDiffusion"]["timesteps"]
        self.logger.info(f"{config['name']}, num_classes: {self.num_classes}, timesteps: {timesteps}")

        ### Configure the U-Net diffusion model for training
        self.device = get_device()  # Auto-detect what device to use for training
        self.amp_dtype = get_amp_dtype(self.device) if self.config["training"]["use_amp"] else None
        self.model = GaussianDiffusion(UNet(**self.config["UNet"]), **self.config["GaussianDiffusion"])
        self.logger.info(f"{self.model.name}: {sum(p.numel() for p in self.model.parameters())} parameters")
        self.model.to(self.device)  # Move this model to the device available

        ### Maintain a model that is an EMA of model weights for stability during sampling
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_decay = 0.999  # This controls the amount of weight that is placed on the prior ema model
        # weights i.e. (1 - ema_decay) is used as the weight on the most recent model weights after another
        # gradient update step has been applied to it

        ### Set up other variables required for training
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.step = 0  # Training step counter, will train until this reaches num_steps
        self.train_losses, self.val_losses = [], []  # Aggregate loss values during training

        ### Create an optimizer and learning rate scheduler
        self.set_config_params(self.config["training"])  # Set training param values as attributes
        self.create_optimizer(self.config["training"])  # Init optimizers with config params
        self.create_lr_scheduler(self.config["training"])  # Init a learning rate scheduler

        ### Load in the most recent checkpoint if specified in the config file
        if self.use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_dir)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training from

    def configure_logging(self):
        """
        Sets up self.logger for logging during training.
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Remove any existing handlers so re-running a notebook cell doesn't duplicate logs
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        # Log to file
        file_handler = logging.FileHandler(os.path.join(self.results_dir, "train.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Log through tqdm
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(formatter)
        self.logger.addHandler(tqdm_handler)
        self.logger.setLevel(logging.INFO)

    def set_config_params(self, train_config: dict) -> None:
        """
        This method extracts relevant parameters from config_dict and sets them as attributes of self.
        e.g. self.batch_size = config_dict["batch_size"].

        :param train_config: A dictionary containing training config parameters.
        """
        defaults = [("batch_size", 64), ("lr_start", 1.0e-3), ("lr_end", 1.0e-5), ("weight_decay", 0.0e-0),
                    ("num_steps", 100000), ("warm_up_pct", 0.05), ("adam_betas", (0.9, 0.999)),
                    ("grad_clip", 1.0), ("use_amp", True), ("use_latest_checkpoint", True),
                    ("eval_every", 1000), ("save_every", 10000)]
        for param_name, default_val in defaults:  # Extract from config dict if possible, otherwise use
            # the default value for each parameter defined immediately above
            default_val = tuple(default_val) if param_name == "adam_betas" else default_val
            setattr(self, param_name, train_config.get(param_name, default_val))

    def create_optimizer(self, train_config: dict) -> None:
        """
        Creates an optimizer for self.model using the parameters recorded in train_config.

        :param train_config: A dictionary containing training config parameters.
        """
        # Configure an optimizer for training self.model, exclude bias and norm layers from weight decay
        exclusions = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                      nn.GroupNorm, nn.LayerNorm, nn.Embedding)
        decay_params, no_decay_params = [], []
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:  # Skip over if no gradient tracking
                    continue

                if isinstance(module, exclusions):
                    # Exclude any kind of batch / group / layer norm from weight decay
                    no_decay_params.append(param)
                elif name == "bias":  # Also exclude any bias terms from weight decay as well
                    no_decay_params.append(param)
                else:  # All others will have weight decay applied to them
                    decay_params.append(param)

        # Check that all params are fully partitioned across decay_params and no_decay_params, check that
        # there is no overlap and also that the total number across both subsets sums to the exp. total
        assert len(set(decay_params).intersection(set(no_decay_params))) == 0
        assert (len(decay_params) + len(no_decay_params) == sum(p.requires_grad
                                                                for p in self.model.parameters()))
        self.opt = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': train_config["weight_decay"]},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=train_config["lr_start"], betas=tuple(train_config["adam_betas"]))

        # Create a grad scaler if needed based on amp_dtype
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None

    def create_lr_scheduler(self, train_config: dict) -> None:
        """
        Creates a learning rate scheduler self.opt using the parameters recorded in train_config.

        :param train_config: A dictionary containing training config parameters.
        """
        # Slowly ramp up the LR from very low to peak with a short warm-up period
        warmup_steps = int(train_config["num_steps"] * train_config["warm_up_pct"])
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Cosine annealing of the learning rate during the rest of training
        decay = CosineAnnealingLR(self.opt, T_max=train_config["num_steps"] - warmup_steps,
                                  eta_min=train_config["lr_end"])
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

    def save(self, milestone: int) -> None:
        """
        Saves the model weights, opt state, lr scheduler state, and loss values for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Writes the weights and losses to disk.
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        if self.scaler is not None:
            data["scaler"] = self.scaler.state_dict()
        torch.save(data, checkpoint_path)

        # Convert the train losses to a pd.DataFrame and save down the results
        df = pd.DataFrame(self.train_losses, columns=["step", "loss"])
        df.to_csv(os.path.join(self.losses_dir, f"train-losses-{milestone}.csv"))

        # Convert the validation losses to a pd.DataFrame and save down the results if any
        if len(self.val_losses) > 0:
            df = pd.DataFrame(self.val_losses, columns=["step", "loss"])
            df.to_csv(os.path.join(self.losses_dir, f"val-losses-{milestone}.csv"))

    def load(self, milestone: int) -> None:
        """
        Loads in the cached model weights, opt state, and lr scheduler state from disk for a particular
        milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. State parameter values are loaded into memory.
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, optimizer state, and lr scheduler state
        # from the checkpoint data read in from disk
        self.step = checkpoint_data["step"]
        self.model.load_state_dict(checkpoint_data["model"])
        self.ema_model.load_state_dict(checkpoint_data["ema_model"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        if self.scaler is not None and "scaler" in checkpoint_data:
            self.scaler.load_state_dict(checkpoint_data["scaler"])
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    @torch.no_grad()
    def update_ema(self) -> None:
        """
        Updates the weights of the ema_model using the current weights in model with a decay parameter
        that specifies how much weight to place on the existing ema_model weights:
            new_ema_wts = curr_ema_wts * (self.ema_decay) + (1 - self.ema_decay) * curr_model_wts
        """
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def report_lr_wd(self):
        """
        Reports the learning rates and weight decay parameter values of the optimizer.
        """
        self.logger.info(f"Reporting learning rates and weight decay at step={self.step}")
        for i, group in enumerate(self.opt.param_groups):  # Report all learning rates
            self.logger.info((f"   lr = {group['lr']:.2e}, wd = {group['weight_decay']:.2e}, "
                              f"count = {len(group['params'])}"))

    def report_memory_usage(self) -> None:
        """
        Reports the current memory usage on the CPU and GPU if available via logging.
        """
        process = psutil.Process(os.getpid())

        cpu_ram_gb = process.memory_info().rss / (1024 ** 3)

        if torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

            self.logger.info(
                f"RAM={cpu_ram_gb:.2f} GB | "
                f"GPU alloc={gpu_alloc_gb:.2f} GB | "
                f"GPU reserved={gpu_reserved_gb:.2f} GB"
            )
        else:
            self.logger.info(f"RAM={cpu_ram_gb:.2f} GB")

    def compute_gradients(self, loss: torch.Tensor) -> None:
        """
        This function essentially performs loss.backward(), but handles using a scaler for certain AMP.

        :param loss: A torch.Tensor with gradient tracking.
        :returns: None, gradients are computed and that information is stored in the optimizer of each model.
        """
        # Compute gradients with a backwards pass using auto-diff
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, model: nn.Module) -> float:
        """
        Performs a gradient update step on the input model, this assumes compute_gradients has already been
        called prior to this method.

        :param model: The model whose parameters are to be updated using the gradients wrt the loss.
        :returns: The grad_norm computed with gradient clipping or np.NaN if no grad clipping is done.
        """
        grad_norm = np.nan  # Set a default value in case self.grad_clip is None
        if self.scaler is not None:
            if self.grad_clip is not None:  # Apply grad clipping if applicable
                self.scaler.unscale_(self.opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)  # Update the model parameters by taking a gradient step
        else:
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.opt.step()  # Update the model parameters by taking a gradient step
        return float(grad_norm)

    @compute_with_amp
    def compute_loss(self, batch: dict) -> Tensor:
        """
        Computes the DDPM MSE training loss over an input batch.

        :param batch: An input batch of data with keys images and class_id.
        :return: A Tensor object that records the MSE loss over this batch.
        """
        x_0 = batch["image"].to(self.device, non_blocking=True)  # Get a batch of clean images (B, 3, H, W)
        class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B,) of integer class ID values
        loss = self.model.training_loss(x_0, class_id)
        return loss

    def train(self) -> None:
        """
        Runs the training of the model until completion for self.num_steps total training iterations.
        """
        msg = f"Starting Training, step={self.step}, device={self.device}, amp_dtype={self.amp_dtype}"
        self.logger.info(msg)
        self.report_lr_wd()  # Report the learning rate and weight decay of the optimizer
        self.model.to(self.device)  # Move the model to the correct device
        self.model.train()  # Set to train for model training
        self.ema_model.to(self.device)  # Move the EMA model used for sampling to the correct device
        self.ema_model.eval()  # Set to eval for sampling training
        self.ema_model.requires_grad_(False)  # Make sure there is no grad tracking for the EMA model

        inf_dataloader = infinite_loader(self.train_dataloader)  # This does not cache batches

        with tqdm(initial=self.step, total=self.num_steps) as pbar:
            while self.step < self.num_steps:  # Run until all training iterations are complete
                batch = next(inf_dataloader)
                loss = self.compute_loss(batch)  # Compute the model loss over this batch with grads
                self.compute_gradients(loss)  # Call backwards() on the loss to compute gradients
                grad_norm = self.optimizer_step(self.model)  # Update model params
                if self.scaler is not None:  # Only call update() iff using this approach
                    self.scaler.update()

                self.scheduler.step()  # Update the learning rate scheduler
                self.update_ema()  # Update the ema_model's weights after this gradient update

                pbar.set_postfix(loss=f"{loss.item():.2f}", grad_norm=f"{grad_norm:.2f}")
                self.train_losses.append((self.step, loss.item()))
                self.step += 1

                ### Periodically run evaluation metrics on the validation data set, always on the last iter
                if self.step % self.eval_every == 0 or self.step == self.num_steps:
                    with torch.no_grad():  # Compute without gradient tracking
                        self.run_eval()

                ### Periodically save the model weights to disk, always on the last iter too
                if self.step % self.save_every == 0 or self.step == self.num_steps:
                    self.save(self.step)
                    # Clear the list of losses after each save, store only the ones from the last save to
                    # the next save
                    self.train_losses, self.val_losses = [], []
                    # Generate new loss plots after saving additional loss data to disk
                    generate_loss_plots(self.losses_dir, self.results_dir)
                    torch.cuda.empty_cache()
                    gc.collect()  # This will slow down training if called too often

                    self.report_lr_wd()  # Report info about the current learning rate and opt state
                    self.report_memory_usage()  # Report info about the memory usage

                del batch, loss, grad_norm
                pbar.update(1)

    @compute_with_amp
    def generate_samples(self, seed: int = None, n_samples: int = 8):
        """
        Generates and saves samples using the EMA model. This method is called periodically in the eval
        routine during training to track the quality of the synthetically generated images during training.
        A grid of images is saved that is (num_classes, n_samples) in size.

        :param seed: An int random seed can be set so that we get similar image generations each eval run
            so that the quality of samples over time can be directly compared.
        :param n_samples: The number of columns in the output saved samples i.e. the number of sample images
            per class to generate and save.
        """
        self.ema_model.eval()  # Switch to eval model for generating samples
        # Create n_samples synthetic images for each class
        class_id = [i for i in range(self.num_classes) for _ in range(n_samples)]
        titles = [f"{i} {self.class_labels[i]}" for i in class_id]
        class_id = torch.tensor(class_id, device=self.device)  # (B = num_classes * n_samples, )
        # Generate fake images i.e. synthetic samples, using the EMA model of prior params
        for cfg_scale in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            # Generate DDIM samples
            x_fake = self.ema_model.ddim_sample(class_id, False, cfg_scale,
                                                self.config["eval"]["sampling_timesteps"],
                                                self.config["eval"]["eta"], seed)
            # Save a copy to both the samples sub-directory and also the main directory to retain the latest
            save_dir = os.path.join(self.samples_dir, f"ddim_cfg_{cfg_scale}")
            os.makedirs(save_dir, exist_ok=True)
            save_images(x_fake, titles, n_samples,
                        os.path.join(save_dir, f"ddim_samples_{self.step}_{cfg_scale}.png"))
            save_images(x_fake, titles, n_samples, os.path.join(self.results_dir,
                                                                f"ddim_samples_latest_{cfg_scale}.png"))

            # Generate DDPM samples
            x_fake = self.ema_model.ddpm_sample(class_id, False, cfg_scale, seed)
            # Save a copy to both the samples sub-directory and also the main directory to retain the latest
            save_dir = os.path.join(self.samples_dir, f"ddpm_cfg_{cfg_scale}")
            os.makedirs(save_dir, exist_ok=True)
            save_images(x_fake, titles, n_samples,
                        os.path.join(save_dir, f"ddpm_samples_{self.step}_{cfg_scale}.png"))
            save_images(x_fake, titles, n_samples, os.path.join(self.results_dir,
                                                                f"ddpm_samples_latest_{cfg_scale}.png"))

    @compute_with_amp
    def run_eval(self):
        """
        This method is used to run periodic model evaluation on the validation set during training.
        """
        was_training = self.model.training
        self.model.eval()
        # 1). Generate fake images i.e. synthetic samples using the ema_model
        self.generate_samples(seed=2026)
        # 2) Compute a MSE loss on the validation set images
        losses, n_obs = [], []
        for batch in self.val_dataloader:
            with torch.no_grad():
                losses.append(self.compute_loss(batch))
            n_obs.append(len(batch["image"]))
        n_obs = torch.tensor(n_obs)
        val_loss = sum(loss * n for loss, n in zip(losses, n_obs)) / n_obs.sum()
        self.val_losses.append((self.step, val_loss.item()))
        self.model.train(was_training)  # Return to training mode to continue training afterwards
