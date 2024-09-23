import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 

# from pytorch3d.loss import chamfer_distance # chamfer distance for calculating point cloud distance
from models.s3dis.mpvcnnpp import MPVCNN2
class feature_map(nn.Module):
  def __init__(self, embed_size, point_size):
    super(feature_map, self).__init__()
    self.embed_size = embed_size
    self.mpvcnn = MPVCNN2(point_size)
    # adding a new last layer
    self.linear = nn.Linear(point_size, embed_size)
    self.batch_norm = nn.BatchNorm1d(embed_size, momentum= 0.01)
  def forward(self, ply):
   
    features = self.mpvcnn(ply)
    features = features.view(features.size(0), -1) # (batch_size, flattened dimensions)
    features = self.batch_norm(self.linear(features)) # (batch_size, embed_size)

    return features