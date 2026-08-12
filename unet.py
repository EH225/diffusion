"""
This module defines the U-Net CNN model used to perform the iterative denoising steps.
"""

import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SinusoidalTimeStepEmb(nn.Module):
    """
    Sinusoidal timestemp embedding for the time steps (t).
    """

    def __init__(self, dim: int):
        """
        Init for the SinusoidalTimeStepEmb layer.

        :param dim: The size of the output (B, dim) returned by the forward pass.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        For an input tensor of size (B, ) this returns a tensor of size (B, dim)
        using sinusoidal timestep embeddings.
        """
        x = x.float()  # Explicitly promote dtype from int to float if necessary
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def UpSample(c_in: int, c_out: int):
    """
    A Conv2d block that up-samples the image feature resolution by a factor of 2.

    :param c_in: The number of input channels i.e. C_in in (B, C_in, H, W).
    :param c_out: The number of output channels i.e. C_out in (B, C_out, H*2, W*2).
    :returns: A nn.Module object instance.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(c_in, c_out, 3, padding=1),
    )


def DownSample(c_in: int, c_out: int) -> nn.Conv2d:
    """
    A nn.Conv2d block that down-samples the image feature resolution by a factor of 2.

    :param c_in: The number of input channels i.e. C_in in (B, C_in, H, W).
    :param c_out: The number of output channels i.e. C_out in (B, C_out, H/2, W/2).
    :returns: A nn.Module object instance.
    """
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)


class SelfAttention(nn.Module):
    """
    Multi-headed self-attention block.
    """

    def __init__(self, c_in: int, groups: int = 32):
        """
        Multi-headed self-attention block between input channels.

        :param c_in: The number of input channels expected.
        :param groups: The number of groups for group norm.
        """
        super().__init__()
        self.norm = nn.GroupNorm(groups, c_in)
        self.self_attention = nn.MultiheadAttention(embed_dim=c_in, num_heads=4, batch_first=True)
        # Define a trainable scaling factor for the residual connection to improve stability
        # self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through multi-headed self-attention block between input channels.

        :param x: An input tensor of size (B, channels, H, W).
        :returns: An output tensor of size (B, channels, H, W).
        """
        B, C, H, W = x.shape  # Extract the shape of the input x tensor
        h = self.norm(x)  # Apply group norm
        h = h.flatten(2).transpose(1, 2)  # Reshape (B, C, H, W) -> (B, H*W, C) -> (B, T, E)
        # Use each spacial location as a separate token, apply multi-headed self-attention to tokens
        h, _ = self.self_attention(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)  # Reshape (B, H*W, C) -> (B, C, H, W)
        # return h * self.res_scale + x  # Add a residual connection to the original input
        return h + x  # Add a residual connection to the original input


class ResnetBlock(nn.Module):
    """
    A ResNet-like block with feature modulation based on FiLM-style conditioning.
    """

    def __init__(self, c_in: int, c_out: int, cond_dim: int, dropout: float = 0.05, groups: int = 32):
        """
        Initializes a ResnetBlock with feature modulation based on FiLM-style conditioning.

        :param c_in: The number of input channels i.e. C_in in (B, C_in, H, W).
        :param c_out: The number of output channels i.e. C_out in (B, C_out, H, W).
        :param cond_dim: The size of the input conditional embedding tensors i.e. (B, cond_dim).
        :param dropout: The amount of dropout to use internally between layers.
        :param groups: The number of channel groups to use for group norm layers.
        """
        super().__init__()
        self.c_in = c_in  # The number of channels coming in (B, c_in, H, W)
        self.c_out = c_out  # The number of channels going out (B, c_out, H, W)
        self.cond_dim = cond_dim  # The size of the context tensor to condition on (B, cond_dim)
        self.drop_prob = dropout

        # This is used to change the dimensions for the resid connection (B, c_in, H, W) -> (B, c_out, H, W)
        # if needed i.e. when there is a difference between c_in and c_out, otherwise no transform needed
        self.res_conv = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        self.norm1 = nn.GroupNorm(min(groups, c_in), c_in)
        self.activation = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)

        # Initialize FiLM layer parameters at mu=0 and sigma^2=1 mimicking the identity transform at first
        self.gamma = nn.Linear(cond_dim, c_out)  # Scale shift
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)

        self.beta = nn.Linear(cond_dim, c_out)  # Mean shift
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

        self.norm2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.activation = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)

        # self.res_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cond_emb: Tensor = None) -> Tensor:
        """
        Forward pass through this convolutional residual block.

        :param x: An input tensor of size (B, c_in, H, W).
        :param cond_emb: A conditional embedding tensor of size (B, cond_dim).
        :return: A tensor of size (B, c_out, H, W).
        """
        x_resid = self.res_conv(x)  # (B, c_in, H, W) -> (B, c_out, H, W) for the residual connection

        x = self.norm1(x)  # Apply group-norm to (B, c_in, H, W)
        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.conv1(x)  # (B, c_in, H, W) -> (B, c_out, H, W)

        x = self.norm2(x)  # Apply group-norm to (B, c_out, H, W)
        # Pass the (B, cond_dim) conditioning tensor through the linear layers to allow the conditioning
        # vector to modulate every feature channel independently, this is the FiLM-style approach
        gamma = self.gamma(cond_emb)[:, :, None, None]  # (B, c_out, 1, 1)
        beta = self.beta(cond_emb)[:, :, None, None]  # (B, c_out, 1, 1)
        x = (1 + gamma) * x + beta  # Apply FiLM-style modulation to incorporate conditional info

        x = self.activation(x)  # Apply SiLU (Sigmoid Linear Unit) activation
        x = self.dropout(x)  # Dropout regularization (B, c_out, H, W) no shape change
        x = self.conv2(x)  # (B, c_out, H, W) -> (B, c_out, H, W)

        # x = x * self.res_scale + x_resid  # Link with x_resid to form a residual connection
        x = x + x_resid  # Link with x_resid to form a residual connection
        return x


class UNet(nn.Module):
    """
    U-Net CNN model consisting of:
        1). An initial Conv2d layer
        2). Multiple down-sampling CNN residual blocks each consisting of:
            [ResnetBlock, ResnetBlock, Downsample]
        3). 2 ResnetBlock middle blocks for the bottleneck
        4). The same number of up-sampling residual CNN blocks each consisting of:
            [Upsample, ResnetBlock, ResnetBlock]
            with skip connections to the earlier down-sampling block outputs
        5). A final Conv2d layer producing an image of the same dimension as the input

    This model can also optionally include attention blocks near the bottleneck for added capacity.
    """

    def __init__(self, image_size: int = 64, num_classes: int = 2, cond_dim: int = 128,
                 dropout: float = 0.05, groups: int = 32, uncond_prob: float = 0.2,
                 use_self_attn: bool = True):
        """
        Instantiates a U-Net model.

        :param image_size: Specifies the size of the input images used during training and the size of the
            output images as well along the spatial dimensions i.e. (B, 3, image_dim, image_dim). This param
            must be in [128, 64, 32].
        :param num_classes: The number of conditional classes, must be an int >= 2.
        :param cond_dim: The dimension of the time and class embedding vectors.
        :param dropout: The amount of dropout regularization to use within the resblocks.
        :param groups: The number of channel groups used in GroupNorm layers.
        :param uncond_prob: Probability of dropping the condition context vector during training.
        :param use_self_attn: If True, then the U-Net model will use self-attention layers internally.
        """
        super().__init__()
        self.name = "UNet"
        assert image_size in [32, 64, 128], "image_size must be one of [32, 64, 128]"
        assert num_classes >= 2 and isinstance(num_classes, int), "Must have at least 2 classes"
        assert isinstance(uncond_prob, float) and 0.0 <= uncond_prob < 1.0
        self.image_size = image_size
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.groups = groups
        # Record the probability of dropping the class_id conditioning vector during training
        self.uncond_prob = uncond_prob
        self.use_self_attn = use_self_attn
        self.dim = 64  # Specify the initial number of channels for the first resblock

        # Define an embedding procedure for the timestamp (t) using an MLP
        self.time_embedding = nn.Sequential(
            SinusoidalTimeStepEmb(self.dim),  # (B,) -> (B, dim)
            nn.Linear(self.dim, self.dim * 4),  # (B, dim) -> (B, 4*dim)
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.cond_dim),  # (B, 4*dim) -> (B, cond_dim)
        )

        # Define an embedding layer for the class_id provided to the model
        # Add 1 extra class embedding for the case of no class i.e. a null category
        self.class_embedding = nn.Embedding(self.num_classes + 1, self.cond_dim)  # (B, cond_dim)

        # 1). An initial convolutional layer which goes from 3 input RGB channels to self.dim
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=3, padding=1)

        # 2). Define the up-sampling blocks and down-sampling blocks
        # We will have the same number of down-sampling blocks as up-sampling blocks, the total number will
        # depend on the input image dimension, the bottleneck will always have a size of (B, C, 4, 4)
        # For each down-sampling block, the channel count will double vs the prior and the output spatial
        # dimensions will be halved

        # For image_size=128 the down-sampling blocks will follow:
        ## TODO: Update these commnets
        # H=W: 128 (img) -> 128 (init_conv) -> (128:64) ->   (64:32) ->   (32:16) ->    (16:8) ->     (8:4)
        # C:     3 (img) ->  64 (init_conv) -> (64:128) -> (128:256) -> (256:512) -> (512:512) -> (512:512)
        # The up-sampling blocks will be the opposite
        # For image_size=64 or 32, the channels end earlier and the down-sampling reaches 4 faster
        last_c = self.dim  # Begin with this as the input number of channels
        c_down_blocks = []  # Record (channels_in, channels_out) for each block
        for _ in range(int(math.log2(image_size) - math.log2(4))):
            c_down_blocks.append((last_c, min(512, last_c * 2)))
            last_c = min(512, last_c * 2)
        c_up_blocks = [(b, a) for a, b in reversed(c_down_blocks)]

        # U-Net down-sampling and up-sampling blocks as a ModuleList of ModuleLists
        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])

        #   A). Create the down-sampling blocks, where each is a ModuleList comprised of 3 sub-blocks:
        # [ResnetBlock, ResnetBlock, Downsample] which operates on dim_in channels and outputs dim_out
        # channels. context_dim is also provided to pass in the context vector
        for idx, (c_in, c_out) in enumerate(c_down_blocks):
            down_block = nn.ModuleList([
                ResnetBlock(c_in, c_in, self.cond_dim, self.dropout, self.groups),
                ResnetBlock(c_in, c_in, self.cond_dim, self.dropout, self.groups),
                (SelfAttention(c_in, self.groups) if (idx >= len(c_down_blocks) - 2) and self.use_self_attn
                 else nn.Identity()),
                DownSample(c_in, c_out),
            ])
            self.downs.append(down_block)

        #   B). Create 2 middle ResNet blocks
        c_mid = c_down_blocks[-1][-1]
        self.mid_block1 = ResnetBlock(c_mid, c_mid, self.cond_dim, self.dropout, self.groups)
        if self.use_self_attn:
            self.mid_attn = SelfAttention(c_mid, self.groups)
        self.mid_block2 = ResnetBlock(c_mid, c_mid, self.cond_dim, self.dropout, self.groups)

        #   C). Create the up-sampling blocks, where each is a ModuleList comprised of 3 sub-blocks:
        # [Upsample, ResnetBlock, ResnetBlock] which operates on dim_in channels and outputs
        # dim_out channels. context_dim is also provided to pass in the context vector.
        for idx, (c_in, c_out) in enumerate(c_up_blocks):
            # To account for the skip connections coming from the encoder down-blocks, the input channels
            # here are x2 larger so that we can concat the outputs from earlier in the network
            up_block = nn.ModuleList([
                UpSample(c_in, c_out),
                ResnetBlock(c_out * 2, c_out, self.cond_dim, self.dropout, self.groups),
                ResnetBlock(c_out * 2, c_out, self.cond_dim, self.dropout, self.groups),
                (SelfAttention(c_out, self.groups) if idx in [0, 1] and self.use_self_attn
                 else nn.Identity()),
            ])
            self.ups.append(up_block)

        # 3). Add 1 final convolution to map to the output channels
        self.out_norm = nn.GroupNorm(self.groups, self.dim)
        self.out_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_channels=self.dim, out_channels=3, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def _cfg_forward(self, x: Tensor, class_id: Tensor, t: Tensor, cfg_scale: float = 3.0) -> Tensor:
        """
        Classifier-free guidance (CFG) forward pass method on an input tensor of noisy images x_t. This method
        outputs a predicted eps or x_0, depending on how the model was trained. CFG runs 2 forward
        passes, one with the class_id condition and one without and combines them to generate a final
        prediction:
            x = (scale + 1) * UNet(x_t, class_id, t) - scale * UNet(x_t, None, t)
            where UNet is the U-Net model forward pass.

        :param x: An input tensor of noisy images of shape (B, C, H, W).
        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param t: An input tensor of shape (B, ) containing the timesteps of each image.
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :returns: An output tensor of shape (B, C, H, W) matching the input x in shape, which is either
            predicted eps or x_0 depending on how the model was trained.
        """
        assert not self.training, "CFG should only be used during evaluation/sampling"
        assert class_id is not None, "class_id cannot be None for CFG forward pass eval"
        # Apply classifier-free guidance using:
        #   x = (scale + 1) * UNet(x_t, class_id, t) - scale * UNet(x_t, None, t)
        x_cond = self.forward(x, class_id, t, 0.0)  # Generate the output x_cond with the class_id context
        x_uncond = self.forward(x, None, t, 0.0)  # Generate again without the class_id context
        x = (cfg_scale + 1) * x_cond - cfg_scale * x_uncond  # Combine into 1 output image
        return x

    def forward(self, x: Tensor, class_id: Tensor, t: Tensor, cfg_scale: float = 0.0) -> Tensor:
        """
        Forward pass through the U-Net model on an input tensor of noisy images x_t. This method outputs a
        predicted eps or x_0, depending on how the model was trained.

        :param x: An input tensor of noisy images of shape (B, C, H, W).
        :param class_id: An input tensor of shape (B, ) containing class IDs for each image.
        :param t: An input tensor of shape (B, ) containing the timesteps of each image.
        :param cfg_scale: A scaling factor used to control how strong the CFG sampling is. Set to 0.0 for
            no CFG sampling at all. 2-5 is usually considered a good range.
        :returns: An output tensor of shape (B, C, H, W) matching the input x in shape, which is either
            predicted eps or x_0 depending on how the model was trained.
        """
        # 0). If cfg_scale > 0, then use the cfg_forward method to compute the forward pass instead
        if cfg_scale > 0:
            return self._cfg_forward(x, class_id, t, cfg_scale)

        # 1). Convert the timestep t into a deep latent vector representation
        time_embed = self.time_embedding(t)

        # 2). Embed the class_id into a deep latent vector representation if one is provided
        if class_id is None:  # Default to a null class embedding vector if no class IDs are provided
            class_id = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
        class_embed = self.class_embedding(class_id)  # (B, cond_dim) class embedding vectors

        # 3). If this is a forward pass called during training, randomly drop the class embedding some of
        # the time as specified by the config parameter uncond_prob
        if self.training:  # Randomly drop the class conditioning
            mask = (torch.rand(class_embed.shape[0]) > self.uncond_prob).float()  # (B, )
            mask = mask[:, None].to(class_embed.device)  # (B, 1) of 1s and 0s
            class_embed = class_embed * mask  # Randomly zero out the class embedding with p=uncond_prob

        # 4). Combine the timestep embedding with the class conditional embedding to obtain 1 context tensor
        cond_emb = time_embed + class_embed  # (B, cond_dim)

        # 5). Run the input x noisy images through the forward pass of the U-Net conditioned on context
        #   A). Initial convolution
        x = self.init_conv(x)

        #   B). Pass the intermediate x through the down-blocks
        resid_conn_features = []  # Use a stack for LIFO processing of the residual connection feature maps
        for down_block in self.downs:  # Iterate over all the down-blocks, which each have 3 component
            x = down_block[0](x, cond_emb)  # ResnetBlock 1
            resid_conn_features.append(x)  # Record this layer's outputs for the residual connection
            x = down_block[1](x, cond_emb)  # ResnetBlock 2
            resid_conn_features.append(x)  # Record this layer's outputs for the residual connection
            if len(down_block) == 4:
                x = down_block[2](x)  # Pass x through a self-attention layer
            x = down_block[-1](x)  # Down-sampling block

        #   C). Pass the intermediate x through the middle blocks i.e. the bottleneck
        x = self.mid_block1(x, cond_emb)
        if self.use_self_attn:
            x = self.mid_attn(x)
        x = self.mid_block2(x, cond_emb)

        #   D). Pass the intermediate x through the up-sampling blocks to restore the origina shape
        for up_block in self.ups:  # Iterate over all the up-blocks, which each have 3 components
            x = up_block[0](x)  # Up-sampling block
            x = up_block[1](torch.concat([x, resid_conn_features.pop()], dim=1), cond_emb)  # ResnetBlock 1
            x = up_block[2](torch.concat([x, resid_conn_features.pop()], dim=1), cond_emb)  # ResnetBlock 2
            if len(up_block) == 4:
                x = up_block[3](x)  # Pass x through a self-attention layer

        #   E). Final conv block
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.final_conv(x)
        return x
