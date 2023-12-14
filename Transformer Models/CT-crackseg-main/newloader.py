import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.598, 0.584, 0.565], [0.104, 0.103, 0.103])
        ])
        return tf(img)

class MaskToTensor(object):
    def __call__(self, mask):
        return torch.from_numpy(mask).long()

# Refactored code here
def load_mask(mask_path, num_classes=7):
    # Load the mask image
    mask_image = Image.open(mask_path)
    
    # Convert mask image to numpy array
    mask_array = np.array(mask_image)
    
    # If mask is RGB, convert it to a unique class map
    if len(mask_array.shape) == 3:  # RGB image
        # Convert RGB to a unique number (assuming 8-bit per channel)
        mask = mask_array[:, :, 0] * 256 * 256 + mask_array[:, :, 1] * 256 + mask_array[:, :, 2]
        
        # Map unique RGB combinations to class indices
        unique_colors = np.unique(mask)
        color_to_class = {color: idx for idx, color in enumerate(unique_colors)}
        mask_class_map = np.vectorize(color_to_class.get)(mask)
        
        # Clip to number of classes
        mask_class_map = np.clip(mask_class_map, 0, num_classes - 1)
        
        return mask_class_map
    else:
        # If mask is already in single channel format, just return it
        return mask_array

class Crack_loader(Dataset):
    def __init__(self, img_dir, img_fnames, mask_dir, mask_fnames, isTrain=False, resize=False, num_classes=8, target_size = (256,256)):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.resize = resize
        self.isTrain = isTrain
        self.num_classes = num_classes
        self.target_size = target_size
        self.aug = A.Compose([
            A.RandomResizedCrop(256, 256, p=0.5),
            A.MotionBlur(p=0.1),
            A.ColorJitter(),
            A.SafeRotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(p=0.5)
        ])
        self.img_totensor = ImgToTensor()
        self.mask_totensor = MaskToTensor()

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = load_mask(mpath, num_classes=self.num_classes)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Use INTER_NEAREST for masks


        if self.isTrain:
            # Apply augmentations first
            transformed = self.aug(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

            # Then convert to tensor
            img = self.img_totensor(Image.fromarray(img))
            mask = self.mask_totensor(mask)

            return {'image': img, 'mask': mask}

        else:
            if self.resize:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                img = self.img_totensor(Image.fromarray(img))

            mask = self.mask_totensor(mask)

            return {'image': img, 'mask': mask, 'img_path': fpath}

    def __len__(self):
        return len(self.img_fnames)