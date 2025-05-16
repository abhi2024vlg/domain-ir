import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np

class LowLightDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, transform=None, augment=True):
        """
        Args:
            root_dir (string): Root directory of the dataset. 
                              Assumes 'high/' and 'low/' subdirectories exist.
            crop_size (int): Size to crop images to (default: 256x256).
            augment (bool): Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.high_dir = os.path.join(root_dir, 'high')
        self.low_dir = os.path.join(root_dir, 'low')
        self.image_filenames = sorted(os.listdir(self.high_dir))  # Assuming paired data
        self.crop_size = crop_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        high_img_path = os.path.join(self.high_dir, self.image_filenames[idx])
        low_img_path = os.path.join(self.low_dir, self.image_filenames[idx])

        high_img = Image.open(high_img_path).convert("RGB")
        low_img = Image.open(low_img_path).convert("RGB")

        # Ensure both images are the same size before cropping
        w, h = high_img.size
        crop_x = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        crop_y = torch.randint(0, h - self.crop_size + 1, (1,)).item()

        high_img = high_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        low_img = low_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))

        # Apply Random Horizontal and Vertical Flip (with 50% probability)
        if self.augment:
            if random.random() > 0.5:  # Horizontal Flip
                high_img = high_img.transpose(Image.FLIP_LEFT_RIGHT)
                low_img = low_img.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:  # Vertical Flip
                high_img = high_img.transpose(Image.FLIP_TOP_BOTTOM)
                low_img = low_img.transpose(Image.FLIP_TOP_BOTTOM)

        # Convert to tensors
        high_img = self.base_transform(high_img)
        low_img = self.base_transform(low_img)

        return low_img, high_img  # Low-light image as input, high-quality as ground truth

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class DeblurringDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, augment=True):
        """
        Args:
            root_dir (string): Root directory of the dataset. 
                              Assumes 'input/' (blurry images) and 'target/' (sharp images) subdirectories.
            crop_size (int): Size to crop images to (default: 256x256).
            augment (bool): Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'input')  # Blurry images
        self.target_dir = os.path.join(root_dir, 'target')  # Sharp images
        self.image_filenames = sorted(os.listdir(self.input_dir))  # Assuming paired data
        self.crop_size = crop_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.image_filenames[idx])
        target_img_path = os.path.join(self.target_dir, self.image_filenames[idx])

        input_img = Image.open(input_img_path).convert("RGB")
        target_img = Image.open(target_img_path).convert("RGB")

        # Ensure both images are the same size before cropping
        w, h = input_img.size
        crop_x = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        crop_y = torch.randint(0, h - self.crop_size + 1, (1,)).item()

        input_img = input_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        target_img = target_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))

        # Apply Random Horizontal and Vertical Flip (with 50% probability)
        if self.augment:
            if random.random() > 0.5:  # Horizontal Flip
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:  # Vertical Flip
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)

        # Convert to tensors
        input_img = self.base_transform(input_img)
        target_img = self.base_transform(target_img)

        return input_img, target_img  # Blurry image as input, sharp image as target

class DerainingDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, augment=True):
        """
        Args:
            root_dir (string): Root directory of the dataset. 
                              Assumes 'input/' (rainy images) and 'target/' (clean images) subdirectories.
            crop_size (int): Size to crop images to (default: 256x256).
            augment (bool): Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'input')  # Rainy images
        self.target_dir = os.path.join(root_dir, 'target')  # Clean images
        self.image_filenames = sorted(os.listdir(self.input_dir))  # Assuming paired data
        self.crop_size = crop_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.image_filenames[idx])
        # Transform the filename for the target directory
        target_filename = self.image_filenames[idx].replace("rain-", "norain-")
        target_img_path = os.path.join(self.target_dir, target_filename)
    
        # Check if files exist
        if not os.path.exists(target_img_path):
            print(f"Warning: Target file not found: {target_img_path}")
            # Skip to next valid image or handle this case however you prefer
            return self.__getitem__((idx + 1) % len(self))
    
        input_img = Image.open(input_img_path).convert("RGB")
        target_img = Image.open(target_img_path).convert("RGB")
    
        # Ensure both images are the same size before cropping
        w, h = input_img.size
    
        # Check if image is smaller than crop size and resize if needed
        if w < self.crop_size or h < self.crop_size:
            # Calculate scale factor needed to make the image large enough
            scale_factor = max(self.crop_size / w, self.crop_size / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        
            # Resize images
            input_img = input_img.resize((new_w, new_h), Image.BILINEAR)
            target_img = target_img.resize((new_w, new_h), Image.BILINEAR)
        
            # Update dimensions
            w, h = new_w, new_h
    
        crop_x = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        crop_y = torch.randint(0, h - self.crop_size + 1, (1,)).item()
    
        input_img = input_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        target_img = target_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
    
        # Rest of your code remains the same
        if self.augment:
            if random.random() > 0.5:  # Horizontal Flip
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:  # Vertical Flip
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Convert to tensors
        input_img = self.base_transform(input_img)
        target_img = self.base_transform(target_img)
    
        return input_img, target_img

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, augment=True):
        """
        Args:
            root_dir (string): Root directory of the dataset. 
                              Assumes 'clean/' (clean images) and 'noisy_15/', 'noisy_25/', 'noisy_50/' subdirectories.
            crop_size (int): Size to crop images to (default: 256x256).
            augment (bool): Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.noisy_dirs = [
            os.path.join(root_dir, 'noisy_15'),
            os.path.join(root_dir, 'noisy_25'),
            os.path.join(root_dir, 'noisy_50')
        ]
        self.image_filenames = sorted(os.listdir(self.clean_dir))  # Assuming paired data
        self.crop_size = crop_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        clean_img_path = os.path.join(self.clean_dir, self.image_filenames[idx])
        noisy_img_paths = [os.path.join(noisy_dir, self.image_filenames[idx]) for noisy_dir in self.noisy_dirs]
        clean_img = Image.open(clean_img_path).convert("RGB")
        noisy_imgs = [Image.open(noisy_img_path).convert("RGB") for noisy_img_path in noisy_img_paths]
    
        # Check if image is smaller than crop size and resize if needed
        w, h = clean_img.size
        if w < self.crop_size or h < self.crop_size:
            # Calculate scale factor needed to make the image large enough
            scale_factor = max(self.crop_size / w, self.crop_size / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        
            # Resize images
            clean_img = clean_img.resize((new_w, new_h), Image.BILINEAR)
            noisy_imgs = [img.resize((new_w, new_h), Image.BILINEAR) for img in noisy_imgs]
        
            # Update dimensions
            w, h = new_w, new_h
    
        # Now perform the random crop - image is guaranteed to be large enough
        crop_x = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        crop_y = torch.randint(0, h - self.crop_size + 1, (1,)).item()
    
        clean_img = clean_img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        noisy_imgs = [img.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size)) for img in noisy_imgs]
    
        # Rest of your code remains the same
        if self.augment:
            if random.random() > 0.5:  # Horizontal Flip
                clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)
                noisy_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in noisy_imgs]
            if random.random() > 0.5:  # Vertical Flip
                clean_img = clean_img.transpose(Image.FLIP_TOP_BOTTOM)
                noisy_imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in noisy_imgs]
            
        # Convert to tensors
        clean_img = self.base_transform(clean_img)
        noisy_imgs = [self.base_transform(img) for img in noisy_imgs]
    
        return noisy_imgs, clean_img
    

# Add this class after the individual dataset definitions
class UnifiedDataset:
    def __init__(self, denoise_ds, derain_ds, deblur_ds, lowlight_ds):
        self.denoise_ds = denoise_ds
        self.derain_ds = derain_ds
        self.deblur_ds = deblur_ds
        self.lowlight_ds = lowlight_ds
        
    def get_random_image_set(self, batch_size=1):
        # Initialize arrays
        img_HR = np.zeros((batch_size, 256, 256, 3))        # Clean images
        img_Rain = np.zeros((batch_size, 256, 256, 3))      # Rainy images
        img_Noise = np.zeros((batch_size, 256, 256, 3))     # Noisy images
        img_Blur = np.zeros((batch_size, 256, 256, 3))      # Blurry images
        img_Lowlight = np.zeros((batch_size, 256, 256, 3))  # Lowlight images
    
        # Fill arrays with actual data
        for i in range(batch_size):
            # Get paired images from each dataset
            noise_idx = random.randint(0, len(self.denoise_ds) - 1)
            rain_idx = random.randint(0, len(self.derain_ds) - 1)
            blur_idx = random.randint(0, len(self.deblur_ds) - 1)
            lowlight_idx = random.randint(0, len(self.lowlight_ds) - 1)
        
            # Get noisy and clean image pair
            noisy_imgs, clean_img_noise = self.denoise_ds[noise_idx]
            img_Noise[i] = noisy_imgs[0].permute(1, 2, 0).numpy()  # Using first noise level
        
            # Get rain and clean image pair
            rain_img, clean_img_rain = self.derain_ds[rain_idx]
            img_Rain[i] = rain_img.permute(1, 2, 0).numpy()
        
            # Get blur and clean image pair
            blur_img, clean_img_blur = self.deblur_ds[blur_idx]
            img_Blur[i] = blur_img.permute(1, 2, 0).numpy()
        
            # Get lowlight and normal light image pair
            lowlight_img, clean_img_lowlight = self.lowlight_ds[lowlight_idx]
            img_Lowlight[i] = lowlight_img.permute(1, 2, 0).numpy()
        
            # Use clean image from denoise dataset for HR
            img_HR[i] = clean_img_noise.permute(1, 2, 0).numpy()
    
        # Return only the needed image types
        return img_HR, img_Noise, img_Rain, img_Blur, img_Lowlight
# class RealWorldEnhanceDataset(Dataset):
#     def __init__(self, root_dir, crop_size=256, augment=True):
#         """
#         Args:
#             root_dir (str):   Root directory containing 'input/' and 'target/' subfolders.
#             crop_size (int):  Size of the square crop. Default is 256.
#             augment (bool):   Whether to apply random flips. Default is True.
#         """
#         self.input_dir  = os.path.join(root_dir, 'input')
#         self.target_dir = os.path.join(root_dir, 'target')
#         self.filenames  = sorted(os.listdir(self.input_dir))  # assumes same names in both
#         self.crop_size  = crop_size
#         self.augment    = augment

#         # Base transform: PIL→Tensor. You can extend with normalization, color jitter, etc.
#         self.base_transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         in_path  = os.path.join(self.input_dir,  self.filenames[idx])
#         tgt_path = os.path.join(self.target_dir, self.filenames[idx])

#         img_in  = Image.open(in_path).convert("RGB")
#         img_tgt = Image.open(tgt_path).convert("RGB")

#         # Random crop
#         w, h = img_in.size
#         if w < self.crop_size or h < self.crop_size:
#             # optional: resize up if too small, or pad
#             img_in  = img_in.resize((self.crop_size, self.crop_size), Image.BICUBIC)
#             img_tgt = img_tgt.resize((self.crop_size, self.crop_size), Image.BICUBIC)
#         else:
#             x = random.randint(0, w - self.crop_size)
#             y = random.randint(0, h - self.crop_size)
#             img_in  = img_in.crop((x, y, x + self.crop_size, y + self.crop_size))
#             img_tgt = img_tgt.crop((x, y, x + self.crop_size, y + self.crop_size))

#         # Random flips
#         if self.augment:
#             if random.random() > 0.5:
#                 img_in  = img_in.transpose(Image.FLIP_LEFT_RIGHT)
#                 img_tgt = img_tgt.transpose(Image.FLIP_LEFT_RIGHT)
#             if random.random() > 0.5:
#                 img_in  = img_in.transpose(Image.FLIP_TOP_BOTTOM)
#                 img_tgt = img_tgt.transpose(Image.FLIP_TOP_BOTTOM)

#         # To tensor
#         img_in  = self.base_transform(img_in)
#         img_tgt = self.base_transform(img_tgt)

#         return img_in, img_tgt
