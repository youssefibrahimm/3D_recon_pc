import sys
# add your path to the sys 
sys.path.insert(0, 'C:\\Users\\youss\\OneDrive - aucegypt.edu\\Youssef\\3D_recon_pc')
# print(sys.path)

from MPVConv.modules.mpvconv import MPVConv
from Transformer_parts import encoderBlock
import torch.nn as nn

class Transpose_layer(nn.Module):
    def forward(self, x):
        """
        Transposes the input tensor along dimensions 1 and 2.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: The transposed tensor of shape (batch_size, sequence_length, channels).
        """
        return x.transpose(1, 2)
    
class AE_ply(nn.Module):
  def __init__(self, latent_size, n_embed, kernel_size, dropout=0.3):
    super(AE_ply, self).__init__()  
    self.latent_size = latent_size
    self.n_embed = n_embed
    self.kernel_size = kernel_size
    self.mpvConv1 = MPVConv(3+latent_size, 64, resolution=32, kernel_size=self.kernel_size)
    self.mpvConv2 = MPVConv(64, 128, resolution=32, kernel_size=self.kernel_size)
    self.mpvConv3 = MPVConv(128, latent_size, resolution=32, kernel_size=self.kernel_size)

    
    self.downsampling_NumPoints = nn.Sequential(
    # input shape (batch_size, n_embed, Num_points)
    Transpose_layer(),
    nn.Linear(self.latent_size, self.n_embed),
    Transpose_layer(),
    nn.Conv1d(in_channels=self.n_embed, out_channels=64, kernel_size=kernel_size, stride=10),
    nn.AvgPool1d(kernel_size=kernel_size, stride=2),
    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=10),
    nn.AvgPool1d(kernel_size=kernel_size, stride=2),
    Transpose_layer(),
    nn.Linear(in_features=64, out_features=n_embed)
    )

    # Encoder block expecting (batch_size, num_points, n_embed)
    self.encoder = encoderBlock(n_embed=self.n_embed, dropout=dropout, num_heads=2)
    # Linear layer to project n_embed back to latent_size
    # self.embed_to_latent = nn.Linear(self.n_embed, self.latent_size)

  def forward(self, input):
    # input should be a tuple containig features (Batch_size, Channel_in, Num_points) and coords (Batch_size, 3, Num_points)
    x, coords = self.mpvConv1(input)
    x, coords = self.mpvConv2((x, coords))
    x, _ = self.mpvConv3((x,coords)) # (batch_size, latent_size, Num_points)
    out_down = self.downsampling_NumPoints(x) # (batch_size, Num_points=11597, n_embed)
    print(f'auto_enc before encoder: {out_down.shape}')
    enc_out, k_enc, v_enc = self.encoder(out_down) # (batch_size, Num_points, n_embed)
    return enc_out, k_enc, v_enc, enc_out.shape[1]
