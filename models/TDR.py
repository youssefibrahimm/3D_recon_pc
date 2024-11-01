from Modules.Dynamic_dec import DynamicDecoder
import torch.functional as F
import torch.nn as nn
from Modules.feature_map import feature_map_AE
class TDR(nn.Module):

    def __init__(self, n_embed, point_size, latent_size, num_heads, min_point_size, kernel_size, width_multiplier, ply_features, stride_conv_upsampling=5, dropout=0.3, loss= nn.MSELoss()):
        super(TDR, self).__init__()
        self.feature_map = feature_map_AE(n_embed=n_embed, kernel_size=kernel_size, 
                                          width_multiplier=width_multiplier, num_points=point_size, dropout=dropout)
        self.dynamic_dec = DynamicDecoder(latent_size=latent_size, 
                                          point_size=point_size,
                                          min_point_size=min_point_size,
                                          num_heads=num_heads,
                                          n_embed=n_embed, 
                                          dropout=dropout,
                                          kernel_size=kernel_size,
                                          ply_features=ply_features,
                                          stride_conv_upsampling=stride_conv_upsampling
                                          )
        self.mse = loss

    def forward(self, x):
        out_features, k_enc, v_enc, first_Dynamic_decoder_linear_layer = self.feature_map(x) # shape out_features: (batch_size, Num_points, embed)
        out = self.dynamic_dec(out_features, k_enc, v_enc, first_Dynamic_decoder_linear_layer) # shape out: (batch_size, ply_features, Num_points)

        # Check if output and input shapes match
        assert out.shape == x.shape, f"Output shape {out.shape} and input shape {x.shape} do not match!"

        # now using MSE for lack of computational power 
        # better use chamfer loss from pytorch3d
       
        loss = self.mse(out, x)

        return out, loss
       