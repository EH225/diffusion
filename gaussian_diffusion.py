"""
This module defines the GaussianDiffusion class which is used to generate images from noise and train the
U-Net model to run a 1-step denoising operation.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm
from typing import Dict, Tuple


class GaussianDiffusion(nn.Module):
    def __init__(self, model, *args, timesteps: int = 100, objective: str = "pred_eps",
                 beta_schedule: str = "cosine"):
        """
        Instantiates a Gaussian Diffusion Model instance.

        Note: All image tensors going in and out are expected to have a range of [-1, +1].

        :param model: A torch model used to run iterative denoising steps on input noisy images.
        :param timesteps: The number of iterative denoising timesteps to use to generate an image i.e. to
            fully decode a pure Gaussian noise start to a clean x_0 image.
        :param objective: Either pred_noise or pred_x_start which defines the training objective of the model
            and what it produces i.e. either the estimated noise added to the original image or the original
            image itself less the noise.
        :param beta_schedule: A beta schedule to use for training i.e. either linear, cosine, or sigmoid.
            This controls the levels of noise used most often during training at each time step. If set to
            linear, then the level of noise increase linearly at each timestep. If cosine, the noise increases
            in a way that is a smooth and gradual increase which can be better suited for DDPM training.
            Sigmoid increases the level of noise more sharply during the middle of the process, this can
            increase the speed of training but may be more prone to instability or sharp transitions.
        """
        super().__init__()
        self.name = "GaussianDiffusion"
        self.model = model  # A model to use for the denoising steps e.g. a U-Net instance
        self.image_size = model.image_size  # The height and width, images are expected to be square
        self.objective = objective  # Specify which objective the model is to predict

        objectives = ["pred_eps", "pred_x_0"]
        assert objective in objectives, f"objective must be one of: {objectives}"

        # Helpful constants are registered below as buffers and can be access through self.name
        # This ensures that they are on the same device as the model parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.Module.html for details
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        ### Noise schedule and alpha values
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        ### Add in other coefficients needed to transform between x_t, x_0 and noise:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise, where noise is sampled from N(0, 1)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        ### Add coeffs for posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the DDPM paper
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer("posterior_mean_coef1",
                        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2",
                        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        ### Add weights for the loss calculation
        snr = alphas_cumprod / (1 - alphas_cumprod)  # Signal-to-noise ratio
        loss_weight = torch.ones_like(snr) if objective == "pred_eps" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, x: Tensor) -> Tensor:
        """
        Maps values [0, 1] to values [-1, 1].
        """
        return x * 2 - 1

    def unnormalize(self, x: Tensor) -> Tensor:
        """
        Maps values [-1, 1] to values [0, 1].
        """
        return (x + 1) * 0.5

    def x_0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """
        Computes x_0 (the original starting image) from x_t (the original img corrupted with noise) given
        the t (the timestep) and eps (the noise added).

        :param x_t: A batch of noise images of shape (B, C, H, W).
        :param t: The timestep of each image in the batch of size (B, ).
        :param eps: A batch of Gaussian noise sampled from N(0, 1) of the same shape as x_t (B, C, H, W).
        :returns: x_0 the batch of starting images corresponding to the batch of noise images, x_t.
        """
        # Transform x_t and noise to get x_0
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # Rearrange the equation: x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*eps
        # where x_0 = x_start, eps = noise
        x_0 = (x_t - sqrt_one_minus_alphas_cumprod * eps) / sqrt_alphas_cumprod
        return x_0

    def eps_from_x_0(self, x_t: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
        """
        Computes the noise implied by x_t (the original img corrupted with noise) and x_0 (the original
        starting image) and t (the timestep).

        :param x_t: A batch of noise images of shape (B, C, H, W).
        :param t: The timestep of each image in the batch of size (B, ).
        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :returns: A batch of noise that was added to x_0 to get x_t the same shape as x_t and x_0.
        """
        # Transform x_t and x_0 to get the noise term
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # Rearrange the equation: x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*eps
        # where x_0 = x_start, eps = noise
        pred_noise = (x_t - sqrt_alphas_cumprod * x_0) / sqrt_one_minus_alphas_cumprod
        return pred_noise

    def q_sample(self, x_0: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """
        Samples from q(x_t | x_0) according to Eq. (4) of the DDPM paper. This creates a noise image from a
        clean one i.e. x_0 by adding noise to it.

        :param x_0: A batch of original images of shape (batch_size, C, H, W).
        :param t: The time step of each image in the batch of size (batch_size, ).
        :param eps: A batch of Gaussian noise sampled from N(0, 1) of the same shape as x_0.
        :returns: A batch of noisy images that are a blend of x_0 clean images and Gaussian noise.
        """
        # q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)I)
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # Sampling from N(mu, sigma^2) can be done as: x_t = mu + sigma * noise where noise ~ N(0, 1)
        mu, sigma = sqrt_alphas_cumprod * x_0, sqrt_one_minus_alphas_cumprod
        x_t = mu + sigma * eps  # (B, C, H, W)
        return x_t

    def training_loss(self, x_0: Tensor, class_id: Tensor) -> Tensor:
        """
        Computes a loss value for training using an input set of original, clean images x_0 and class_id
        that serves as class-conditional context. The following process is used:
            1). Randomly sample timesteps t from [0, self.num_timesteps] for each x_0 image
            2). Randomly sample Gaussian noise for each x_0 image
            3). Generate an x_t using x_0 and the noise for each obs in the batch
            4). Pass x_t and t into the model and predict either the noise that was added or x_0
            5). Compare the model's prediction (i.e. the U-Net output) vs the ground truth

        :param x_0: A batch of original images of shape (B, C, H, W). Note, these are expected to already
            be normalized to [-1, 1] when passed to this function.
        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :returns: A single torch float representing the loss from running a training iteration for 1 batch.
        """
        # x_0 = self.normalize(x_0)  # (B, C, H, W) convert [0, 1] to [-1, 1] values
        assert (-1.0 <= x_0).all() and (x_0 <= 1.0).all()
        min_val = x_0.min()
        if min_val >= 0:
            print(f"x_0.min() >= 0, {min_val}, input x_0 may not be properly scaled to [-1, 1]")

        B = x_0.shape[0]  # Batch size
        # t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()  # (B,) random t
        # 50% of batches/examples sample uniformly
        t_unif = torch.randint(0, self.num_timesteps, (B,), device=x_0.device).long()  # (B,)
        # 50% explicitly sample low-noise timesteps
        t_low = torch.randint(0, int(self.num_timesteps * 0.2), (B,), device=x_0.device).long()  # (B,)
        mask = torch.rand(B, device=x_0.device) < 0.5
        t = torch.where(mask, t_low, t_unif)  # Sample timesteps more so on the lower end where the model
        # makes the most errors and where the errors most impact the visual quality of the outputs

        eps = torch.randn_like(x_0)  # (B, C, H, W) create Gaussian noise N(0, 1) of the same shape
        target = eps if self.objective == "pred_eps" else x_0  # (B, C, H, W)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (B, C, H, W)
        # Implements the loss function according to Eq. (14) of the DDPM paper
        # Sample x_t from q(x_t | x_0) using the `q_sample` function
        x_t = self.q_sample(x_0, t, eps)  # Generate a noisy image using the starting image
        # Compute the y-hat values, will either be x_0 or eps, but will match target from above
        y_hat = self.model(x_t, class_id, t)
        loss = (torch.pow(target - y_hat, 2) * loss_weight).mean()  # Compute the weighted MSE Loss
        return loss

    def q_posterior(self, x_0: Tensor, x_t: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns the posterior mean and stddev i.e. q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of
        the DDPM paper. This is used in p_sample (not training). This is a helper method for ddpm_sample.

        :param x_0: A batch of original images of shape (B, C, H, W).
        :param x_t: A batch of noise images of shape (B, C, H, W).
        :param t: The timestep of each image in the batch of size (B, ).
        :returns:
            posterior_mean: (B, C, H, W) tensor. Mean of the posterior.
            posterior_std: (B, C, H, W) tensor. Std of the posterior.
        """
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_0 + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, class_id: Tensor, t: int, cfg_scale: float = 3.0,
                 seed: int = None) -> Tensor:
        """
        Samples from p(x_{t-1} | x_t) according to Eq. (6) of the DDPM paper. This returns 1 step forward
        of the de-noising process i.e. x_{t-1} is 1 step less noisy than x_t with x_0 being a clean image.
        This method is used in the DDPM sampling process during inference (not training). This is a helper
         method for ddpm_sample.

        :param x_t: A batch of noise images of shape (B, C, H, W).
        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param t: An integer denoting the denoising timestep currently being run. Note this is a single int
            and not a tensor of ints, it's the same int used for all images in the batch.
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :param seed: A random seed that can be set to make sampling repeatable.
        :returns: A batch of images x_{t-1} that are 1 step less noisy, same size and shape as x_t.
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (B,) of all the same val t
        # sample x_{t-1} from p(x_{t-1} | x_t)
        # Get the model's prediction, note the model can predict either x_0 or the noise
        if self.objective == "pred_x_0":  # The model output will be the predicted x_0
            x_0 = self.model(x_t, class_id, t, cfg_scale)
        elif self.objective == "pred_eps":  # The model output will be the predicted noise
            eps = self.model(x_t, class_id, t, cfg_scale)
            x_0 = self.x_0_from_eps(x_t, t, eps)  # Convert from eps to x_0 using epx and x_t
        x_0 = x_0.clamp(-1, 1)  # Clamp to the valid range [-1, 1] to ensure the generate remains stable

        # Get the mean and std for q(x_{t-1} | x_t, x_0) using self.q_posterior, and sample x_{t-1}
        posterior_mean, posterior_std = self.q_posterior(x_0, x_t, t)
        rng = torch.Generator(device=class_id.device)  # Get up a random number generator
        if seed is not None:  # Set the seed if one is provided for replicability
            rng.manual_seed(seed)
        # Generate Gaussian noise N(0, 1) of size (B, C, H, W)
        noise = torch.randn_like(x_t, device=x_t.device, generator=rng)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # Handle if t == 0, then no noisy sampling
        x_tm1 = posterior_mean + nonzero_mask * posterior_std * noise
        return x_tm1

    @torch.no_grad()
    def ddpm_sample(self, class_id: Tensor, return_all_t: bool = False, cfg_scale: float = 3.0,
                    seed: int = None) -> Tensor:
        """
        This method uses the slower DDPM sampling approach, which visits all timesteps.

        Generates a batch of generated images of size (B, C, H, W) given the input class_id, which will
        provide class ID values as context. This method begins with B=batch_size pure Gaussian noise images of
        size (B, C, H, W) and applies a series of iterative denoising operations to them and returns the clean
        images when finished with values [0, 1]. This method is used in inference (not training).

        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param return_all_t: If set to True, then the first (all noise) and all T denoising timestep images
            are returned as a tensor of size (batch_size, T+1, C, H, W). Otherwise, just the last image
            is returned i.e. the maximally denoised one of size (B, C, H, W).
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :param seed: A random seed that can be set to make sampling repeatable.
        :returns: A tensor of denoised images of size:
                (B, T+1, C, H, W) if return_all_t is True else (B, C, H, W)
                with values [-1, +1].
        """
        self.eval()  # Set to eval mode for inference, switch off dropout and effects batch norm
        device = class_id.device
        img_shape = (len(class_id), 3, self.image_size, self.image_size)  # (B, C, H, W)
        rng = torch.Generator(device=device)  # Get up a random number generator
        if seed is not None:  # Set the seed if one is provided for replicability
            rng.manual_seed(seed)
        x_t = torch.randn(img_shape, device=device, generator=rng)  # Generate pure noise ~ N(0, 1)
        # Create a list to hold the images that are denoised, starting with a pure noise image
        x_t_all = [x_t] if return_all_t else None

        for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM sampling", total=self.num_timesteps):
            # Iteratively apply denoising steps to the image to move towards an original, clean image x_0
            x_t = self.p_sample(x_t, class_id, t, cfg_scale, seed + t)
            if return_all_t:  # Only record the intermediate image steps if specified
                x_t_all.append(x_t)

        res = torch.stack(x_t_all, dim=1) if return_all_t else x_t
        # res = self.unnormalize(res)  # Res has values [-1, 1] due to clamping, map to [0, 1] instead
        return res

    def get_ddim_sigma(self, t_int: int, t_int_prev: int, eta: float) -> Tensor:
        """
        Returns the sigma_t value associated with the timestep t. This is a helper method for ddim_sample.

        This is computed as:
            sigma_t = eta * sqrt((1-alpha_bar_t_prev)/(1-alpha_bar_t)) * sqrt(1 - alpha_t/alpha_t_prev)

        :param t_int: The current timestep as an int.
        :param t_int_prev: The next sampling timestep / previous noise process timestep where
            t_int > t_int_prev since we count down from T (most noise) to 0 (no noise).
        :param eta: Controls how much additional random noise is injected during each DDIM step. Set to 0
            for deterministic sampling.
        :return: The sigma_t value used in DDIM sampling.
        """
        if eta == 0:
            return torch.zeros((), device=self.betas.device)
        alpha_bar_t = self.alphas_cumprod[t_int]
        alpha_bar_prev = self.alphas_cumprod[t_int_prev]
        A = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
        B = torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        return eta * A * B

    @torch.no_grad()
    def ddim_step(self, x_t: Tensor, clas_id: Tensor, t_int: int, t_int_prev: int, eta: float,
                  cfg_scale: float = 3.0, seed: int = None) -> Tensor:
        """
        This is a helper method for ddim_sample that return x_{t-k} from a given input x_t. The DDIM sampling
        method takes larger size k steps than the DDPM sampling method which always takes size 1 steps.

        This method computes:

            x_{t-1} = sqrt(alpha_bar_t_prev)*x_0 + sqrt(1 - alpha_bar_t_prev - sigma_t^2)*eps + sigma_t*z

        where z = N(0, 1) and the same size as x_{t-1}. This added noise term is 0 if eta=0.

        :param x_t: An input tensor of noisy images of shape (B, C, H, W).
        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param t_int: The current timestep as an int.
        :param t_int_prev: The next sampling timestep / previous noise process timestep where
            t_int > t_int_prev since we count down from T (most noise) to 0 (no noise).
        :param eta: Controls how much additional random noise is injected during each DDIM step. Set to 0
            for deterministic sampling.
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :param seed: A random seed that can be set to make sampling repeatable.
        :return: A batch of images x_{t-1} that are 1 step less noisy, same size and shape as x_t.
        """
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)  # (B,) of t_int
        # Get the model's prediction, note the model can predict either x_0 or the noise
        if self.objective == "pred_x_0":  # The model output will be the predicted x_0
            x_0 = self.model(x_t, clas_id, t, cfg_scale)
            eps = self.eps_from_x_0(x_t, t, x_0)  # Convert from x_0 to eps using x_0 and x_t
        elif self.objective == "pred_eps":  # The model output will be the predicted noise
            eps = self.model(x_t, clas_id, t, cfg_scale)
            x_0 = self.x_0_from_eps(x_t, t, eps)  # Convert from eps to x_0 using eps and x_t+
        x_0 = x_0.clamp(-1, 1)  # Clamp to the valid range [-1, 1] to ensure the generate remains stable

        if t_int_prev < 0:  # If we're at the final DDIM sampling step, just return x_0, no noise to be added
            return x_0

        # Add some noise along the way during DDIM sampling to get diverse images
        sigma = self.get_ddim_sigma(t_int, t_int_prev, eta)  # DDIM noise amount
        alpha_bar_prev = self.alphas_cumprod[t_int_prev]
        # Direction pointing toward x_t from x_0
        pred_direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps
        # Compute the DDIM update i.e. x_t -> x_{t-1}
        rng = torch.Generator(device=x_t.device)  # Get up a random number generator
        if seed is not None:  # Set the seed if one is provided for replicability
            rng.manual_seed(seed)
        noise = torch.randn_like(x_t, device=x_t.device, generator=rng)
        x_tmk = torch.sqrt(alpha_bar_prev) * x_0 + pred_direction + sigma * noise
        return x_tmk  # x_{t-k} (B, C, H, W)

    @torch.no_grad()
    def ddim_sample(self, class_id: Tensor, return_all_t: bool = False, cfg_scale: float = 3.0,
                    sampling_timesteps: int = 50, eta: float = 0.0, seed: int = None) -> Tensor:
        """
        This method uses the faster DDIM sampling approach, which visits only a few timesteps.

        Generates a batch of generated images of size (B, C, H, W) given the input class_id, which will
        provide class ID values as context for each generated image. This method begins with B=batch_size
        pure Gaussian noise images of size (B, C, H, W) and applies a series of iterative denoising operations
        to them and returns the clean images when finished with values [0, 1]. This method is used in
        inference (not training).

        The model is trained with self.num_timesteps diffusion steps, but sampling can use a smaller number
        of sampling_timesteps instead.

        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param return_all_t: If set to True, then the first (all noise) and all T denoising timestep images
            are returned as a tensor of size (batch_size, T+1, C, H, W). Otherwise, just the last image
            is returned i.e. the maximally denoised one of size (B, C, H, W).
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :param sampling_timesteps: The number of sampling timesteps to use in DDIM sampling.
        :param eta: Controls how much additional random noise is injected during each DDIM step. Set to 0
            for deterministic sampling.
        :param seed: A random seed that can be set to make sampling repeatable.
        :returns: A tensor of denoised image of size either:
                (B, sampling_timesteps+1, C, H, W) if return_all_t is True else (B, C, H, W)
                with values [-1, +1].
        """
        msg = "sampling_timesteps must be less than self.num_timesteps"
        assert sampling_timesteps <= self.num_timesteps, msg
        assert sampling_timesteps > 0, "sampling_timesteps must be greather than 0"
        assert 0.0 <= eta <= 1.0, "eta must be 0.0 <= eta <= 1.0"

        self.eval()  # Set to eval mode for inference, switch off dropout and effects batch norm
        device = self.betas.device

        # Create a subset of the training timesteps to use during sampling
        # Example: 1000 training steps -> 50 DDIM sampling steps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, sampling_timesteps, device=device).long()

        # Start from pure Gaussian noise x_T
        img_shape = (len(class_id), 3, self.image_size, self.image_size)  # (B, C, H, W)
        rng = torch.Generator(device=device)  # Get up a random number generator
        if seed is not None:  # Set the seed if one is provided for replicability
            rng.manual_seed(seed)
        x_t = torch.randn(img_shape, device=self.betas.device, generator=rng)  # Generate pure noise ~ N(0, 1)
        # Create a list to hold the images that are denoised, starting with a pure noise image
        x_t_all = [x_t] if return_all_t else None

        for i in tqdm(range(sampling_timesteps), desc="DDIM sampling", total=sampling_timesteps):
            # Iteratively apply denoising steps to the image to move towards an original, clean image x_0
            t_int = timesteps[i]  # Get the current timestemp x_t that we're currently at
            # Determine what the timestep will be, use -1 to represent the final x_0 output
            # t_int_prev is the next timestep in the DDIM sampling process and prev step in the forward
            # noising process from x_0 -> x_T which is why it is called prev
            t_int_prev = timesteps[i + 1] if i < sampling_timesteps - 1 else -1
            x_t = self.ddim_step(x_t, class_id, t_int, t_int_prev, eta, cfg_scale, seed + i)
            if return_all_t:  # Only record the intermediate image steps if specified
                x_t_all.append(x_t)

        res = torch.stack(x_t_all, dim=1) if return_all_t else x_t
        # res = self.unnormalize(res)  # Res has values [-1, 1] due to clamping, map to [0, 1] instead
        return res


########################
### Helper Functions ###
########################

def extract(cache: Tensor, t: Tensor, x_shape: Tuple[int]) -> Tensor:
    """
    Extracts the appropriate coefficient values based on the given timesteps.

    This function gathers the values from the coefficient tensor cache according to the given timesteps t
    and reshapes them to match the required shape such that it supports broadcasting with the tensor of
    given shape x_shape.

    :param cache: A tensor of shape (T,), containing coefficient values for all timesteps.
    :param t: A tensor of shape (b,), representing the timesteps for each sample in the batch.
    :param x_shape: The shape of the input image tensor, usually (B, C, H, W).
    :returns: A tensor of shape (B, 1, 1, 1), containing the extracted coefficient values from a for
        corresponding timestep of each batch element, reshaped accordingly.
    """
    B, *_ = t.shape  # Extract batch size from the timestep tensor
    out = cache.gather(-1, t)  # Gather the coefficient values from cache based on t
    out = out.reshape(B, *((1,) * (len(x_shape) - 1)))  # Reshape to (b, 1, 1, 1) for broadcasting
    return out


def linear_beta_schedule(timesteps: int) -> Tensor:
    """
    Computes a linear schedule of beta values proposed in original DDPM paper.

    :param timesteps: The total number of timesteps to create beta values for.
    :returns: A Tensor of beta values, one for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Computes a cosine schedule of beta values proposed in Improved Denoising Diffusion Probabilistic Models
    (https://openreview.net/forum?id=-NEXDKk8gZ).

    :param timesteps: The total number of timesteps to create beta values for.
    :returns: A Tensor of beta values, one for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Computes a sigmoid schedule of beta values proposed in Scalable Adaptive Computation for Iterative
    Generation (https://arxiv.org/abs/2212.11972). Figure 8 suggets that this schedule is better for images
    of size 64 x 64 during training.

    :param timesteps: The total number of timesteps to create beta values for.
    :returns: A Tensor of beta values, one for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule: str, timesteps: int) -> Tensor:
    """
    Computes a schedule of beta values, one for each timestep and returns them as a Tensor.

    :param beta_schedule: Specifies which type of schedule to use i.e. linear, cosine, or sigmoid.
    :param timesteps: The number of iterative denoising timesteps to use to generate an image i.e. to
        fully decode a pure Gaussian noise start to a clean x_0 image.
    :returns: A Tensor of beta values, one for each timestep.
    """
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    # Data validation checks on output betas
    assert torch.all(betas > 0)
    assert torch.all(betas < 1)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    assert torch.all(alphas_cumprod[1:] < alphas_cumprod[:-1])
    return betas
