from __future__ import division
import copy
import torch.optim as optim
from utils.utils import *
from pathlib import Path
from newloader import Crack_loader
from model.TransMUNet import TransMUNet
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setup_seed(42)
number_classes = 8
input_channels = 3
best_val_loss = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify your data paths as strings
data_tra_path = r'C:\Users\skoka3\Desktop\CIVE-FP\test_train_valid\train'
data_val_path = r'C:\Users\skoka3\Desktop\CIVE-FP\test_train_valid\val'

DIR_IMG_tra = os.path.join(data_tra_path, 'images')
DIR_MASK_tra = os.path.join(data_tra_path, 'masks')
DIR_IMG_val = os.path.join(data_val_path, 'images')
DIR_MASK_val = os.path.join(data_val_path, 'masks')

# Specify your hyperparameters
batch_size_tr = 4  # Specify your desired batch size for training
batch_size_va = 4  # Specify your desired batch size for validation
lr = 0.001  # Specify your desired learning rate
epochs = 20  # Specify your desired number of epochs
saved_model = 'finetuned_Crackformer.pth'  # Specify the path to save the model
saved_model_final = 'finetuned_Crackformer_final.pth'  # Specify the path to save the final model
patience = 6  # Specify the patience value
progress_p = 0.5  # Specify the progress percentage value
pretrained = 0  # Specify whether to use pretrained weights (0 or 1)
loss_filename = 'loss_log.txt'  # Specify the loss log file name
save_result = './results/'  # Specify the path to save results

# Load data file names
img_names_tra = [path.name for path in Path(DIR_IMG_tra).glob('*.jpg')]
mask_names_tra = [path.name for path in Path(DIR_MASK_tra).glob('*.png')]

img_names_val = [path.name for path in Path(DIR_IMG_val).glob('*.jpg')]
mask_names_val = [path.name for path in Path(DIR_MASK_val).glob('*.png')]

train_dataset = Crack_loader(img_dir=DIR_IMG_tra, img_fnames=img_names_tra, mask_dir=DIR_MASK_tra,
                              mask_fnames=mask_names_tra, isTrain=True)
valid_dataset = Crack_loader(img_dir=DIR_IMG_val, img_fnames=img_names_val, mask_dir=DIR_MASK_val,
                              mask_fnames=mask_names_val, resize=True)

print(f'train_dataset:{len(train_dataset)}')
print(f'valiant_dataset:{len(valid_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size_tr, shuffle=True, drop_last=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size_va, shuffle=False, drop_last=True)

Net = TransMUNet(n_classes=number_classes)
flops, params = get_model_complexity_info(Net, (3, 512, 512), as_strings=True, print_per_layer_stat=False)
print('flops: ', flops, 'params: ', params)
message = 'flops:%s, params:%s' % (flops, params)

Net = Net.to(device)

# Load pretrained model if specified
if pretrained == 1:
    Net.load_state_dict(torch.load(saved_model, map_location='cpu')['model_weights'])
    best_val_loss = torch.load(saved_model, map_location='cpu')['val_loss']

optimizer = optim.Adam(Net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)
criteria = Custom_DiceBCELoss()

# Initialize visualizer and log file
visualizer = Visualizer(isTrain=True)
log_name = os.path.join('./checkpoints', loss_filename)
with open(log_name, "a") as log_file:
    log_file.write('%s\n' % message)
# Lists to store losses
train_losses = []
val_losses = []

# Training loop
for ep in range(epochs):
    Net.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {ep + 1}/{epochs}")

    for itter, batch in enumerate(train_loader_tqdm):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device, dtype=torch.long)  # Ensure mask is long type
        msk_pred, B = Net(img, istrain=True)

        loss = criteria(msk_pred, msk)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {ep + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

    # Validation
    with torch.no_grad():
        Net.eval()
        val_loss = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Validation Epoch {ep + 1}/{epochs}")

        for itter, batch in enumerate(val_loader_tqdm):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device, dtype=torch.long)
            msk_pred = Net(img)

            loss = criteria(msk_pred, msk)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {ep + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')

        # Check the performance and save the model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'New best validation loss: {best_val_loss:.4f}, saving model...')
            torch.save({'model_weights': Net.state_dict(), 'val_loss': best_val_loss}, saved_model)

# Save the final model
torch.save({'model_weights': Net.state_dict(), 'val_loss': best_val_loss}, saved_model_final)

# Save the loss lists for plotting
np.save('train_losses.npy', np.array(train_losses))
np.save('val_losses.npy', np.array(val_losses))