import torch.nn as nn
import torch.nn.functional as F
from Modules.Transformer_parts import MultiHeadAttention, DecoderBlock

class DynamicDecoder(nn.Module):
    def __init__(self, latent_size, point_size, max_point_size, num_heads, n_embed, dropout=0.3):  
        """
        Initialization of the DynamicDecoder.
        
        :param latent_size: Number of features in the latent representation.
        :param point_size: Number of points in the original point cloud.
        :param max_point_size: Maximum number of points the decoder should output.
        :param num_heads: Number of attention heads.
        :param n_embed: Number of features in each point.
        """ 
        super(DynamicDecoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        self.max_point_size = max_point_size  # Maximum points after upsampling
        
        # Fully connected layers for initial reconstruction
        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, max_point_size * 3)  # Output max number of points

        self.embed_to_latent = nn.Linear(self.n_embed, self.latent_size)
        # self.latent_to_embed = nn.Linear(self.latent_size, self.n_embed),

        # Further processing with feedforward layers and decoder blocks
        self.decoder_block = DecoderBlock(num_heads=num_heads, dropout=dropout, n_embed=n_embed)

    def forward(self, embeds, k_encoder, v_encoder):
        """
        Forward pass for the dynamic decoder.

        :param embeds: Embedded representation of the input point cloud (from auto-encoder)
        :param k_encoder: Encoder's keys (from multi-head attention in encoder)
        :param v_encoder: Encoder's values (from multi-head attention in encoder)
        """
        # Use the attention results with the decoder block
        attentive_points = self.decoder_block(embeds, k_encoder, v_encoder)
        
        # Transform the attentive points back to the latent space
        latent_att_points = self.embed_to_latent(attentive_points)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(latent_att_points))
        x = F.relu(self.fc2(x))
        points = self.fc3(x)  # Get max number of points
        
        points = points.view(-1, 3, self.max_point_size)  # Reshape to (batch_size, 3, max_point_size)

        # Select final points dynamically based on point_size
        selected_points = points[:, :, :self.point_size]  # Dynamic point selection shape: (batch_size, 3, point_size)

        return selected_points

