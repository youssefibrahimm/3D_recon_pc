import torch.functional as F
import torch.nn as nn
from Modules.Dynamic_dec import DynamicDecoder
from Modules.feature_map import feature_map_AE
class TDR(nn.Module):

    def __init__(self, n_embed, point_size, latent_size, num_of_feat, num_heads, max_point_size, kernel_size, width_multiplier, isConv=True):
        super(TDR, self).__init__()
        self.feature_map = feature_map_AE(latent_size=latent_size, num_of_feat=num_of_feat, n_embed=n_embed, kernel_size=kernel_size, width_multiplier=width_multiplier, num_points=point_size, isConv=isConv)
        self.dynamic_dec = DynamicDecoder(latent_size=latent_size, 
                                          point_size=point_size,
                                          max_point_size=max_point_size,
                                          num_heads=num_heads,
                                          n_embed=n_embed
                                          )
        self.mse = nn.MSELoss()

    def forward(self, x):
        out_features, k_enc, v_enc = self.feature_map(x) # shape out_features: (batch_size, Num_points, latent_size)
        out = self.dynamic_dec(out_features, k_enc, v_enc) # shape out: (batch_size, Num_points, 3)

        out = out.permute(0, 2, 1) # shape out: (batch_size, 3, Num_points)
        # Check if output and input shapes match
        assert out.shape == x.shape, f"Output shape {out.shape} and input shape {x.shape} do not match!"

        # now using MSE for lack of computational power 
        # better use chamfer loss from pytorch3d
       
        loss = self.mse(out, x)

        return out, loss
       