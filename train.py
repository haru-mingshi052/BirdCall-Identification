import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from seed_everything import seed_everything
from models import Model
from dataset import BCIDataset

import warnings

import argparse

parser = argparse.ArgumentParser(
    description = "data preprocessing"
)

parser.add_argument("--data_folder", type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
parser.add_argument('--es_patience', default = 15, type = int,
                    help = "どれだけ改善が無い場合学習を止めるか")
parser.add_argument('--epochs', default = 100, type = int,
                    help = "何エポック学習するか")
args = parser.parse_args()

#=============================
# split
#=============================
def split():
    df = pd.read_csv(args.data_folder + '/sample_df.csv')

    target = df['bird']
    target = pd.get_dummies(target)

    return train_test_split(df, target, random_state = 71, test_size = 0.2)

#==============================
# train
#==============================
def train():
    tr_x, val_x, tr_y, val_y = split()

    #基礎モデル
    model = EfficientNet.from_pretrained('efficientnet-b1')

    model = Model(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス：', device)
    
    tr_ds = BCIDataset(tr_x, tr_y, args.data_folder)
    val_ds = BCIDataset(val_x, val_y, args.data_folder)
    
    tr_dl = DataLoader(tr_ds, batch_size = 64, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = 64, shuffle = False)
    
    patience = args.es_patience
    best_loss = np.inf
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = 'min', 
        patience = 3, 
        verbose = True, 
        factor = 0.2
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        val_loss = 0
        train_correct = 0
        val_correct = 0
        
        for x, y in tr_dl:
            x = torch.tensor(x, device = device, dtype = torch.float32)
            y = torch.tensor(y, device = device, dtype = torch.float32)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(z, dim = 1)
            target = torch.argmax(y, dim = 1)
            train_correct += (pred.cpu() == target.cpu()).sum()
            train_loss += loss.item()
        train_acc = train_correct.detach().numpy() / len(tr_ds)
        train_loss_list.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_dl:
                x_val = torch.tensor(x_val, device = device, dtype = torch.float32)
                y_val = torch.tensor(y_val, device = device, dtype = torch.float32)
                z_val = model(x_val)
                loss = criterion(z_val, y_val)
                val_pred = torch.argmax(z, dim = 1)
                val_target = torch.argmax(y, dim = 1)
                val_correct += (val_pred.cpu() == val_target.cpu()).sum()
                val_loss += loss.item()
            val_acc = val_correct.detach().numpy() / len(val_ds)
            val_loss_list.append(val_loss)
            
            finish_time = time.time()
            
            print('Epochs：{:03} | Loss：{:.5f} | acc：{:.3f} | Val Loss：{:.5f} | Val acc：{:.3f} | Time：{:.3f}'
                 .format(epoch, train_loss, train_acc, val_loss, val_acc, finish_time - start_time))
            
            scheduler.step(val_loss)
            
            if val_loss <= best_loss:
                best_epoch = epoch
                best_train_loss = train_loss
                best_loss = val_loss
                best_train_acc = train_acc
                best_val_acc = val_acc
                patience = args.es_patience

            else:
                patience -= 1
                if patience == 0:
                    print('Early stooping | Epochs：{:03} | Loss：{:.5f} | acc：{:.3f} | Val Loss：{:.5f} | Val acc：{:.3f}'
                          .format(epoch, train_loss, train_acc, best_loss, val_acc))
                    break

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train()