from MPVConv.modules.mpvconv import MPVConv
from Modules.Transformer_parts import encoderBlock
import torch.nn as nn
import torch

class Encoder(nn.Module):
  def __init__(
      self,
      latent_size: int,
      kernel_size: int,
      point_size: int,
      dropout: float = 0.3,
      resolution: int = 32,
  ):
    """
    Initialize the encoder module.

    Parameters:
    latent_size (int): The size of the latent space.
    n_embed (int): The number of features in the input point cloud.
    kernel_size (int): The size of the kernel used in the MPVCNN layers.
    point_size (int): The number of points in the input point cloud.
    dropout (float, optional): The dropout probability. Defaults to 0.3.
    """
    super().__init__()

    # Check that the input parameters are not None
    if latent_size is None:
      raise ValueError("latent_size is null")
    if kernel_size is None:
      raise ValueError("kernel_size is null")
    if point_size is None:
      raise ValueError("point_size is null")
    if dropout is None:
      raise ValueError("dropout is null")

    # Save the input parameters
    self.latent_size = latent_size
    self.kernel_size = kernel_size
    self.point_size = point_size
    self.dropout = dropout

    # Initialize the MPVCNN layers
    self.mpv_conv_1 = MPVConv(
      3,  # input channel
      64,  # output channel
      resolution=resolution,  # resolution of the input point cloud
      kernel_size=kernel_size
    )
    self.mpv_conv_2 = MPVConv(
      64,  # input channel
      128,  # output channel
      resolution=resolution,  # resolution of the input point cloud
      kernel_size=kernel_size
    )
    self.mpv_conv_3 = MPVConv(
      128,  # input channel
      latent_size,  # output channel
      resolution=resolution,  # resolution of the input point cloud
      kernel_size=kernel_size
    )

  def forward(self, input):
    # input should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    ply, coord = input[:,:3,:].contiguous(), input[:,:3,:].contiguous()
    x, coords = self.mpv_conv_1((ply, coord ))
    x, coords = self.mpv_conv_2((x, coords))
    x, coords = self.mpv_conv_3((x,coords)) # X: (batch_size, latent_size, Num_points)
    xMax = torch.max(x, 2, keepdim=True)[0]
    return xMax.squeeze(2), coords
