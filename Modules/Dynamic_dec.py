import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.Transformer_parts import MultiHeadAttention, DecoderBlock
from Modules.ply_autoenc import Transpose_layer
class DynamicDecoder(nn.Module):
    def __init__(self, latent_size, point_size, max_point_size, num_heads, n_embed, kernel_size, ply_features ,dropout=0.3):  
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
        self.n_embed = n_embed
        self.ply_features = ply_features
        
        # Fully connected layers for initial reconstruction
        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, max_point_size * 3)  # Output max number of points

        self.embed_to_latent = nn.Linear(self.n_embed, self.latent_size)
        # self.latent_to_embed = nn.Linear(self.latent_size, self.n_embed),

        # Further processing with feedforward layers and decoder blocks
        self.decoder_block = DecoderBlock(num_heads=num_heads, dropout=dropout, n_embed=n_embed)

        self.convtranspose_upsampling = nn.Sequential(
        # input shape (batch_size, n_embed, Num_points)
        # nn.Linear(self.latent_size, self.n_embed),
        Transpose_layer(),
        nn.ConvTranspose1d(in_channels=self.n_embed, out_channels=64, kernel_size=kernel_size, stride=5),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        Transpose_layer(),
        nn.Linear(in_features=64, out_features=latent_size),
        Transpose_layer()
        )   
    
        self.ply_features = nn.Sequential(
            nn.Linear(latent_size, self.ply_features),
            Transpose_layer(),
            nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        )
    def forward(self, embeds, k_encoder, v_encoder):
        """
        Forward pass for the dynamic decoder.

        :param embeds: Embedded representation of the input point cloud (from auto-encoder)
        :param k_encoder: Encoder's keys (from multi-head attention in encoder)
        :param v_encoder: Encoder's values (from multi-head attention in encoder)
        """
        # Use the attention results with the decoder block
        attentive_points = self.decoder_block(embeds, k_encoder, v_encoder)

        upsample_conv = self.convtranspose_upsampling(attentive_points)
        print(f'upsampled conv shape: {upsample_conv.shape}')
        # Transform the attentive points back to the latent space
        latent_att_points = self.embed_to_latent(attentive_points)
        print("latent_att_points shape: ", latent_att_points.shape)
        # Pass through fully connected layers
        x = F.relu(self.fc1(latent_att_points))
        x = F.relu(self.fc2(x))
        upsample_linear = self.fc3(x)  # Get max number of points
       
        # Ensure upsample_linear shape is as expected
        assert upsample_linear.shape[2] == self.max_point_size * 3, f"Expected output shape of {self.max_point_size * 3}, but got {upsample_linear.shape[2]}."

        combined_upsampling = torch.cat((upsample_conv, upsample_linear), dim=-1) # output shape: (batch_size, latent_size, conv_up_points + linear_up_points)
        print(f'combined_upsampling shape: {combined_upsampling.shape}')
        combined_upsampling.transpose(1,2) # output shape: (batch_size, conv_up_points + linear_up_points, latent_size)

        downsampled_avgPooled_points = self.ply_features(combined_upsampling) # output shape: (batch_size, ply_features, downsampled_points)

        selected_points = downsampled_avgPooled_points.view(downsampled_avgPooled_points.shape[0], self.ply_features, self.point_size)  # Reshape to (batch_size, 3+additional_feattures, max_point_size)

        return selected_points

