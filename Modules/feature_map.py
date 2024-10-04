# import sys
# # add your path to the sys 
# sys.path.insert(0, 'C:\\Users\\youss\\OneDrive - aucegypt.edu\\Youssef\\3D_recon_pc')
# # print(sys.path)

import torch.nn as nn
import torch
from Modules.MPVConv.models.s3dis.mpvcnnpp import MPVCNN2
from ply_autoenc import AE_ply


class feature_map_AE(nn.Module):
  def __init__(self, latent_size, num_of_feat, n_embed, width_multiplier):
    super(feature_map_AE, self).__init__()
    self.latent_size = latent_size
    self.mpvcnn = MPVCNN2(num_of_feat, width_multiplier=width_multiplier)
    # adding a new last layer
    self.linear = nn.Linear(num_of_feat*width_multiplier, latent_size)
    self.layer_norm = nn.LayerNorm(latent_size)
    self.Auto_enc = AE_ply(latent_size=latent_size, n_embed=n_embed)

  def feat_map(self, ply):
    # ply should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    features, coords = ply, ply[:,:3,:].contiguous()
    features_1 = self.mpvcnn(features) # (batch_size, num_of_feat, num_of_points)
    print(f'feature_2 after mean:{features_1.shape}')
    features_2 = features_1.mean(dim=2) # (batch_size, num_of_feat)

    # adjusting the shape of the output to pass to the autoencoder
    # Normalize and project to latent_size
    print(f'feature_2 after mean:{ features_2.shape}')
    linear_layer = self.linear(features_2)
    print(f'linear_layer:{ linear_layer.shape}')
    feat_3 = self.layer_norm(linear_layer).unsqueeze(1) # (batch_size, latent_size, 1)
    print(f'feat_3:{feat_3.shape}')
    features_3 = feat_3.expand(-1, -1, features_1.size(2)) # (batch_size, latent_size, num_of_points)
    print(f'features_3_expand:{features_3.shape}')

    # Combine the coordinates with the features
    coords_expanded = coords.unsqueeze(1).expand(-1, 3, -1) # (batch_size, 3, num_of_points)
    features_3 = torch.cat([coords_expanded, features_3], dim=1)  # (batch_size, 3+latent_size, num_of_points)
    return features_3, coords_expanded 
  
  def forward(self, ply):
   feat_map, coords = self.feat_map(ply)
   x, k_enc, v_enc = self.Auto_enc((feat_map, coords))
   return x, k_enc, v_enc
   