import torch.functional as F
import torch.nn as nn
from Modules.feature_map import feature_map_AE
from Modules.Dynamic_dec import DynamicDecoder

class TDR(nn.Module):

    def __init__(self, n_embed, point_size, latent_size, head_size, num_of_feat, num_heads, max_point_size):
        super(TDR, self).__init__()
        self.n_embded = n_embed
        self.point_size = point_size
        self.latent_size = latent_size
        self.head_size = head_size
        self.num_of_feat = num_of_feat
        self.num_heads = num_heads
        self.max_point_size = max_point_size
        self.feature_map = feature_map_AE(latent_size=latent_size, num_of_feat=num_of_feat)
        self.dynamic_dec = DynamicDecoder(latent_size=latent_size, 
                                          point_size=point_size,
                                          max_point_size=max_point_size,
                                          num_heads=num_heads,
                                          n_embed=n_embed,
                                          head_size=head_size)
        

    def forward(self, x):
        out_features, k_enc, v_enc = self.feature_map(x) # shape out_features: (batch_size, Num_points, latent_size)
        out = self.dynamic_dec(out_features, k_enc, v_enc) # shape out: (batch_size, Num_points, 3)

        # now using MSE for lack of computational power 
        # better use chamfer loss from pytorch3d
        mse = nn.MSELoss
        loss = mse(out, x)

        return out, loss
        
