import torch.nn as nn
import torch
from Modules.MPVConv.models.s3dis.mpvcnnpp import MPVCNN2
from Modules.Transformer_parts import encoderBlock
from Modules.ply_autoenc import Transpose_layer
# from ply_autoenc import AE_ply


class feature_map_AE(nn.Module):
  """
  Initializes the feature_map_AE module.

  Parameters:
      n_embed (int): The embedding size.
      kernel_size (int): The size of the kernel used in the MPVCNN2 layers.
      width_multiplier (int): The width multiplier.
      num_points (int): The number of points in the input point cloud.
      dropout (float, optional): The dropout probability. Defaults to 0.3.

  Attributes:
      num_points (int): The number of points in the input point cloud.
      mpvcnn (MPVCNN2): The MPVCNN2 instance.
      downsampling_NumPoints (nn.Sequential): The downsampling layer.
      encoder (encoderBlock): The encoder block.

  """
  def __init__(self, n_embed, kernel_size, width_multiplier, num_points, dropout=0.3):
    super(feature_map_AE, self).__init__()
    assert kernel_size>=3, f"kernel_size shoud be at least 3, but got {kernel_size}"
    self.num_points = num_points
    self.mpvcnn = MPVCNN2(n_embed, width_multiplier=width_multiplier)

    self.downsampling_NumPoints = nn.Sequential(
        # input shape (batch_size, n_embed, Num_points)
        nn.Conv1d(in_channels=n_embed, out_channels=64, kernel_size=kernel_size, stride=10),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=10),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        Transpose_layer(),
        nn.Linear(in_features=64, out_features=n_embed),
    )

    self.encoder = encoderBlock(n_embed=n_embed, dropout=dropout, num_heads=2)


  def forward(self, ply):
    # ply should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    features_1, coords = self.mpvcnn(ply) # (batch_size, num_of_feat, num_of_points)
    features_2 = self.downsampling_NumPoints(features_1)
    out_feat_encoder, k_enc, v_enc = self.encoder(features_2) # (batch_size, num_of_points, n_embed)
    return out_feat_encoder, k_enc, v_enc, coords

