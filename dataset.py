import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class BCIDataset(Dataset):
    def __init__(self, df, target, im_folder):
        self.df = df
        self.target = target
        self.im_folder = im_folder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = np.array(Image.open(self.im_folder + '/output/' + self.df.iloc[index].song_sample))
        img = img.transpose(2,0,1)
        
        y = self.target.iloc[index]
        y = np.array(y, dtype = np.float32)
        
        return img, y