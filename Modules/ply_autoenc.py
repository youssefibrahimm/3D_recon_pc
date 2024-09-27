import sys
# add your path to the sys 
sys.path.insert(0, 'C:\\Users\\youss\\OneDrive - aucegypt.edu\\Youssef\\3D_recon_pc')
# print(sys.path)

from MPVConv.modules.mpvconv import MPVConv
from Transformer_parts import encoderBlock
import torch.nn as nn
class AE_ply(nn.Module):
  def __init__(self, latent_size, n_embed, head_size):
    super(AE_ply, self).__init__()  
    self.latent_size = latent_size
    self.n_embed = n_embed
    self.mpvConv1 = MPVConv(3, 64, resolution=32, kernel_size=1)
    self.mpvConv2 = MPVConv(64, 128, resolution=32, kernel_size=1)
    self.mpvConv3 = MPVConv(128, latent_size, resolution=32, kernel_size=1)

    # Linear layer to project latent_size to n_embed
    self.latent_to_embed = nn.Linear(latent_size, n_embed)

    # Encoder block expecting (batch_size, num_points, n_embed)
    self.encoder = encoderBlock(n_embed=n_embed, latent_size=latent_size, num_heads=2, head_size=head_size )
    # Linear layer to project n_embed back to latent_size
    self.embed_to_latent = nn.Linear(n_embed, latent_size)

  def forward(self, input):
    # input should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    x, coords = self.mpvConv1(input)
    x, coords = self.mpvConv2((x, coords))
    x, _ = self.mpvConv3((x,coords)) # (batch_size, latent_size, Num_points)
    x = x.transpose(1,2) # (batch_size, Num_points, latent_size)
    x = self.latent_to_embed(x) # (batch_size, Num_points, n_embed)
    x, k_enc, v_enc = self.encoder(x) # (batch_size, Num_points, n_embed)
    x = self.embed_to_latent(x) # (batch_size, Num_points, latent_size)
    return x, k_enc, v_enc
