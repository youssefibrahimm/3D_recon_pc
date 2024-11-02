import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.MPVConv.models.s3dis.mpvcnnpp import MPVCNN2
from Modules.Encoder import Encoder
from Modules.Decoder import Decoder

class PointCloudAutoencoder(nn.Module):
  def __init__(
      self,
      num_input_features: int,
      num_latent_features: int,
      kernel_size: int,
      num_points: int,
      dropout_probability: float = 0.3,
      hidden_sizes_decoder: list[int] = [1024, 2048, 4096], 
      hidden_sizes_AE: list[int] = [128, 256, 512]

  ):
    """
    Initialize a PointCloudAutoencoder model.

    Args:
        num_input_features (int): The number of features in the input point cloud.
        num_latent_features (int): The size of the latent space.
        num_embed_features (int): The number of features in the embedded point cloud.
        kernel_size (int): The size of the kernel used in the MPVCNN2 layers.
        num_points (int): The number of points in the input point cloud.
        dropout_probability (float, optional): The probability of dropout in the layers. Defaults to 0.3.
    """
    super(PointCloudAutoencoder, self).__init__()
    assert num_latent_features >= hidden_sizes_AE[-1], f'Last layer must have at least as many neurons as the latent space, but got {hidden_sizes_AE[-1]} and {num_latent_features}'
   
    self.latent_size = num_latent_features
    self.kernel_size = kernel_size
    self.point_size = num_points
    self.num_feat = num_input_features
    
    self.dropout = nn.Dropout(dropout_probability)
    self.decoder = Decoder(num_latent_features, num_points, dropout=dropout_probability, hidden_sizes=hidden_sizes_decoder)
    self.encoder = Encoder(num_latent_features, kernel_size, num_points, dropout_probability)
    self.mpvcnnpp = MPVCNN2(num_input_features)

    sizes = [num_input_features] + hidden_sizes_AE + [num_latent_features]
    layers = []
    for i in range(len(sizes) - 1):
      layers.append(nn.Linear(sizes[i], sizes[i+1]))
      layers.append(self.dropout)
      layers.append(nn.ReLU())
    self.AE = nn.Sequential(*layers)

    self.weighted_fusion_linear = nn.Linear(num_points, num_points)

  def forward(self, input_points: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the PointCloudAutoencoder model.

    Args:
        input_points (torch.Tensor): The input point cloud, with shape (batch_size, num_points, 3 + num_features).

    Returns:
        torch.Tensor: The output point cloud, with shape (batch_size, num_points, 3 + num_features).
    """
    assert self.point_size <= input_points.size(2), f"The parameter point_size: {self.point_size} is greater than the points in the ply: {input_points.size(2)} "
    input_points = input_points[:, :, :self.point_size].contiguous()

    # Encode the input point cloud
    encoded_points, _ = self.encoder(input_points)

    # Compute features with MPVCNN2
    features, _ = self.mpvcnnpp(input_points)

    # Apply average pooling over the point_size dimension
    pooled_features = F.adaptive_avg_pool1d(features, 1)  # Output shape will be (batch_size, num_feat, 1)

    # Flatten the features and compute the latent reconstruction
    flattened_features = features.view(pooled_features.size(0), -1)
    latent_reconstruction = self.AE(flattened_features)

    # Decode the latent reconstruction with the two decoders
    reconstruction_mpv = self.decoder(latent_reconstruction)
    reconstruction_enc = self.decoder(encoded_points)

    # Compute the enhanced attention mechanism
    combined_features = torch.cat([reconstruction_mpv, reconstruction_enc], dim=2).transpose(2, 1)
    combined_features = self.dropout(combined_features)
    attention_weights = self.weighted_fusion_linear(combined_features)
    attention_weights = attention_weights.transpose(2, 1) @ combined_features * (combined_features.size(2) ** -0.5)
    attention_weights = F.softmax(attention_weights, dim=1)
    attention_weights = self.dropout(attention_weights)

    # Compute the weighted sum of coordinates based on attention weights
    output_points = torch.bmm(attention_weights, combined_features.transpose(2,1)).transpose(2, 1)
    return output_points
