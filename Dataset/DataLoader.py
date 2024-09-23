from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
import numpy as np
class Read_ply(Dataset):
    def __init__(self, ply_path):
        super(Read_ply, self).__init__()
        self.ply = ply_path
        # self.points = np.asarray(o3d.io.read_point_cloud(self.ply).points).reshape(3,-1)
    
    def __len__(self):
        return len(self.ply)
    
    def __getitem__(self, index):
        points = np.asarray(o3d.io.read_point_cloud(self.ply[index]).points).reshape(3,-1) 
        return points, points

    def collate__fn(self, points):
        features = np.stack(p[0] for p in points) # (B, 3, N)
        coordintates = np.stack(p[1] for p in points) # (B, 3, N)
        return features, coordintates  

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)


def GetDataLoader(ply_path, batch_size, train_set_percentage=0.9,shuffle=True, drop_last=True):
    # Defining the dataset
    ds = Read_ply(ply_path)
    
    # Randomly splitting the dataset
    train_set, test_set = RandomSplit(ds, train_set_percentage)

    # Defining the dataloader
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=ds.collate__fn)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=ds.collate__fn)
    
    return train_dl, test_dl

    