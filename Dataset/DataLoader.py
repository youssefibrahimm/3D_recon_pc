from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
import numpy as np
import torch
class Read_ply(Dataset):
    def __init__(self, ply_path):
        super(Read_ply, self).__init__()
        self.ply = ply_path
    
    def __len__(self):
        return len(self.ply)
    
    def __getitem__(self, index):
        points = np.asarray(o3d.io.read_point_cloud(self.ply[index]).points).reshape(3,-1) 
        return points, points

    def collate__fn(self, points):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features = np.stack([p[0] for p in points], axis = 0) # (B, 3, N)
        coordintates = np.stack([p[1] for p in points], axis = 0) # (B, 3, N)
        f = torch.tensor(features).float().to(device)
        c = torch.tensor(coordintates).float().to(device)
        pad_channels = 9 - f.size(1) - c.size(1)  # Calculate how many extra channels are needed
        padding = torch.zeros(f.size(0), pad_channels, f.size(2), device=device)  # Create padding tensor
        data = torch.cat((f, c, padding), dim=1).to(device)
        return data  

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)


def GetDataLoader(ply_path, batch_size, train_set_percentage=0.9, shuffle=True, drop_last=True):
    # Defining the dataset
    ds = Read_ply(ply_path)
    
    # Randomly splitting the dataset
    train_set, test_set = RandomSplit(ds, train_set_percentage)

    # Defining the dataloader
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=ds.collate__fn)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=ds.collate__fn)
    
    return train_dl, test_dl
