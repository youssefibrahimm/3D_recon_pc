# import sys
# # add your path to the sys 
# sys.path.insert(0, 'C:\\Users\\youss\\OneDrive - aucegypt.edu\\Youssef\\3D_recon_pc')
# # print(sys.path)

import torch.nn as nn
import torch
from Modules.MPVConv.models.s3dis.mpvcnnpp import MPVCNN2
from ply_autoenc import AE_ply


class feature_map_AE(nn.Module):
  """
  Initializes the feature_map_AE module.

  Parameters:
      latent_size (int): The size of the latent space.
      num_of_feat (int): The number of features.
      n_embed (int): The embedding size.
      width_multiplier (int): The width multiplier.
      num_points (int): The number of points in the input point cloud.
      isConv (bool): Whether to use convolutional layers for projection to latent_size, or linear layers.

  Attributes:
      latent_size (int): The size of the latent space.
      mpvcnn (MPVCNN2): The MPVCNN2 instance.
      linear (nn.Linear): Linear layer for the last layer.
      layer_norm (nn.LayerNorm): Layer normalization instance.
      Auto_enc (AE_ply): The AE_ply instance.

  """
  def __init__(self, latent_size, num_of_feat, n_embed, kernel_size, width_multiplier, num_points, coordinates=3, isConv=True, dropout=0.3):
    super(feature_map_AE, self).__init__()
    assert kernel_size>=3, f"kernel_size shoud be at least 3, but got {kernel_size}"

    self.latent_size = latent_size
    self.num_points = num_points
    self.isConv = isConv
    self.mpvcnn = MPVCNN2(num_of_feat, width_multiplier=width_multiplier)

    # adding a new last layer
    self.linear = nn.Linear(num_of_feat*width_multiplier, latent_size)
    self.conv1x1 = nn.Conv1d(num_of_feat*width_multiplier, out_channels=1, kernel_size=kernel_size, stride=(num_points//self.latent_size))
    self.layer_norm = nn.LayerNorm(latent_size)
    self.Auto_enc = AE_ply(latent_size=latent_size, n_embed=n_embed, kernel_size=kernel_size, dropout=dropout)

  def feat_map(self, ply):
    # ply should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    # features, coords = ply, ply[:,:3,:].contiguous()
    features_1, coords = self.mpvcnn(ply) # (batch_size, num_of_feat, num_of_points)
    print(f'feature_2 before :{features_1.shape}')
    print(f'coords shape: {coords.shape}')
    if self.isConv:
      features_1 = self.conv1x1(features_1).squeeze(1) # (batch_size, latent-size)
      print(f'conv layer output shape :{features_1.shape}')
    else:
      pred_points=[]
      for feat_idx in range(features_1.shape[1]):
        feat_pred = features_1[:,feat_idx,:]
        pred_points.append(feat_pred) 

      features_2 = torch.mean(torch.stack(pred_points, dim=1), dim=2) # (batch_size, num_of_feat)
      # project to latent_size
      linear_layer = self.linear(features_2)
      print(f'linear_layer:{ linear_layer.shape}')

    feat_3 = self.layer_norm(linear_layer).unsqueeze(2) # (batch_size, latent_size, 1)
    print(f'feat_3:{feat_3.shape}')

    features_3 = feat_3.expand(-1, -1, features_1.size(2)) # (batch_size, latent_size, num_of_points)
    print(f'features_3_expand:{features_3.shape}')

    # Combine the coordinates with the features
    # coords_expanded = coords.unsqueeze(1).expand(-1, 3, -1) # (batch_size, 3, num_of_points)
    features_3 = torch.cat([coords, features_3], dim=1)  # (batch_size, 3+latent_size, num_of_points)
    print(f'features_3_cat:{features_3.shape}')
    return features_3, coords 
  
  def forward(self, ply):
   feat_map, coords = self.feat_map(ply)
   x, k_enc, v_enc, first_Dynamic_decoder_linear_layer = self.Auto_enc((feat_map, coords))
   return x, k_enc, v_enc, first_Dynamic_decoder_linear_layer
   