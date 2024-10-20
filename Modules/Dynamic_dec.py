import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.Transformer_parts import MultiHeadAttention, DecoderBlock
from Modules.ply_autoenc import Transpose_layer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DynamicDecoder(nn.Module):
    def __init__(self, latent_size, point_size, min_point_size, num_heads, n_embed, kernel_size, ply_features, stride_conv_upsampling=5, dropout=0.3):  
        """
        Initialization of the DynamicDecoder.
        
        :param latent_size: Number of features in the latent representation.
        :param point_size: Number of points in the original point cloud.
        :param min_point_size: Minimum number of points the decoder should output.
        :param num_heads: Number of attention heads.
        :param n_embed: Number of features in each point.
        """ 
        super(DynamicDecoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        self.min_point_size = min_point_size  
        self.n_embed = n_embed
        self.ply_features = ply_features
        

        self.embed_to_latent = nn.Linear(self.n_embed, self.latent_size)
        # self.latent_to_embed = nn.Linear(self.latent_size, self.n_embed),

        # Further processing with feedforward layers and decoder blocks
        self.decoder_block = DecoderBlock(num_heads=num_heads, dropout=dropout, n_embed=n_embed)

        self.convtranspose_upsampling = nn.Sequential(
        # input shape (batch_size, n_embed, Num_points)
        # nn.Linear(self.latent_size, self.n_embed),
        Transpose_layer(),
        nn.ConvTranspose1d(in_channels=self.n_embed, out_channels=64, kernel_size=kernel_size, stride=stride_conv_upsampling),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        nn.ReLU(),
        Transpose_layer(),
        nn.Linear(in_features=64, out_features=latent_size),
        Transpose_layer()
        )   
    
        self.ply_features = nn.Sequential(
            nn.Linear(latent_size, self.ply_features),
            Transpose_layer(),
            nn.AvgPool1d(kernel_size=kernel_size, stride=2),
        )
    def forward(self, embeds, k_encoder, v_encoder, first_Dynamic_decoder_linear_layer=None):
        """
        Forward pass for the dynamic decoder.

        :param embeds: Embedded representation of the input point cloud (from auto-encoder)
        :param k_encoder: Encoder's keys (from multi-head attention in encoder)
        :param v_encoder: Encoder's values (from multi-head attention in encoder)
        """
        # Fully connected layers for initial reconstruction
        self.fc1 = nn.Linear(first_Dynamic_decoder_linear_layer, 1024, device=device)
        self.fc2 = nn.Linear(1024, 2048, device=device) 
        self.fc3 = nn.Linear(2048, self.min_point_size * 2, device=device)  
        print(f"embeds: {embeds.shape}")
        # Use the attention results with the decoder block
        attentive_points = self.decoder_block(embeds, k_encoder, v_encoder)

        upsample_conv = self.convtranspose_upsampling(attentive_points)
        print(f'upsampled conv shape: {upsample_conv.shape}')
        # Transform the attentive points back to the latent space
        latent_att_points = self.embed_to_latent(attentive_points).transpose(1,2)
        print("latent_att_points shape: ", latent_att_points.shape)
        # Pass through fully connected layers
        x = nn.ReLU()(self.fc1(latent_att_points))
        x = nn.ReLU()(self.fc2(x))
        upsample_linear = self.fc3(x)  # Get max number of points
        print(f'linear upsample: {upsample_linear.shape}')
        # Ensure upsample_linear shape is as expected
        assert upsample_linear.shape[2] == self.min_point_size * 2, f"Expected output shape of {self.min_point_size * 2}, but got {upsample_linear.shape[2]}."
        

        combined_upsampling = torch.cat((upsample_conv, upsample_linear), dim=-1) # output shape: (batch_size, latent_size, conv_up_points + linear_up_points)
        combined_upsampling=combined_upsampling.transpose(1,2) # output shape: (batch_size, conv_up_points + linear_up_points, latent_size)
        print(f'combined_upsampling shape: {combined_upsampling.shape}')

        downsampled_avgPooled_points = self.ply_features(combined_upsampling) # output shape: (batch_size, ply_features, downsampled_points)
        print(f'down shape: {downsampled_avgPooled_points.shape}')
        to_point_size = nn.Linear(downsampled_avgPooled_points.shape[2], self.point_size, device=device)
        selected_points = to_point_size(downsampled_avgPooled_points) 
        print(f'selected_points: {selected_points.shape}')

        return selected_points

