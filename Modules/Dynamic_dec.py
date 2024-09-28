import sys
# add your path to the sys 
sys.path.insert(0, 'C:\\Users\\youss\\OneDrive - aucegypt.edu\\Youssef\\3D_recon_pc')
# print(sys.path)


import torch.nn as nn
import torch.nn.functional as F
from Modules.Transformer_parts import MultiHeadAttention, DecoderBlock
class DynamicDecoder(nn.Module):
    def __init__(self, latent_size, point_size, max_point_size, num_heads, n_embed):
        super(DynamicDecoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        self.max_point_size = max_point_size  # Maximum points after upsampling
        
        # Fully connected layers for initial reconstruction
        self.fc1 = nn.Linear(latent_size, 512)
        print("fc1 weight shape:", self.fc1.weight.shape)
        self.fc2 = nn.Linear(512, 512)
        print("fc2 weight shape:", self.fc2.weight.shape)
        self.fc3 = nn.Linear(512, max_point_size * 3)  # Output max number of points
        print("fc3 weight shape:", self.fc3.weight.shape)

        # Multi-Head Attention for dynamic selection
        self.attention_layer = MultiHeadAttention(num_heads=num_heads, n_embed=n_embed, decoder=False)

        # Further processing with feedforward layers and decoder blocks
        self.decoder_block = DecoderBlock(num_heads=num_heads, n_embed=n_embed)

    def forward(self, latent, k_encoder, v_encoder):
        """
        Forward pass for the dynamic decoder.
        :param latent: Latent representation of the input point cloud (from auto-encoder)
        :param k_encoder: Encoder's keys (from multi-head attention in encoder)
        :param v_encoder: Encoder's values (from multi-head attention in encoder)
        """
        # Pass through fully connected layers
        x = F.relu(self.fc1(latent))
        x = F.relu(self.fc2(x))
        points = self.fc3(x)  # Get max number of points
        points = points.view(-1, self.max_point_size, 3)  # Reshape to (batch_size, max_point_size, 3)

        # Attention mechanism to focus on key points
        attention_output, _, _, _ = self.attention_layer(points)

        # Use the attention results with the decoder block
        x = self.decoder_block(attention_output, k_encoder, v_encoder)

        # Select final points dynamically based on point_size
        selected_points = x[:, :self.point_size, :]  # Dynamic point selection shape: (batch_size, point_size, 3)


        return selected_points

