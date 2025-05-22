import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms 
import numpy as np
from typing import Callable, List
from PIL import Image
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from RandStainNA.randstainna import RandStainNA


class WSIBagDatasetMIL(Dataset):
    def __init__(self,
                 slide_list_csv: str,
                 patches_root_dir: str, # Root directory where WSI-specific patch folders are located
                 label_column: str = "label",
                 slide_id_column: str = "slide_id",
                 model_input_size: int = 384,
                 patch_file_extensions: tuple = ('.png', '.jpg', '.jpeg', '.tif', '.tiff') # Supported patch image extensions
                 ):
        self.patches_root_dir = patches_root_dir
        self.label_column = label_column
        self.slide_id_column = slide_id_column
        self.model_input_size = model_input_size
        self.patch_file_extensions = patch_file_extensions

        try:
            self.slide_df = pd.read_csv(slide_list_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Slide list CSV not found at {slide_list_csv}")
        except Exception as e:
            raise ValueError(f"Error reading slide list CSV {slide_list_csv}: {e}")

        if self.slide_id_column not in self.slide_df.columns:
            raise ValueError(f"Slide ID column '{self.slide_id_column}' not found in {slide_list_csv}")
        if self.label_column not in self.slide_df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in {slide_list_csv}")

        self.slide_data = []
        print(f"Scanning for slides listed in {slide_list_csv}...")
        for _, row in self.slide_df.iterrows():
            slide_id_from_csv = str(row[self.slide_id_column]) # Ensure slide_id is a string
            label = row[self.label_column]

            base_slide_id, _ = os.path.splitext(slide_id_from_csv)
            wsi_patch_folder = os.path.join(self.patches_root_dir, base_slide_id)
            if os.path.isdir(wsi_patch_folder):
                # Get all image files, sort them for consistency if needed
                patch_files = sorted([
                    os.path.join(wsi_patch_folder, f) 
                    for f in os.listdir(wsi_patch_folder) 
                    if f.lower().endswith(self.patch_file_extensions)
                ])
                if patch_files:
                    self.slide_data.append({
                        'slide_id': base_slide_id, 
                        'label': label, 
                        'patch_paths': patch_files # Store full paths to patch images
                    })
                else:
                    print(f"Warning: No patch images found in folder: {wsi_patch_folder} for slide_id {base_slide_id} (from CSV: {slide_id_from_csv}). Skipping.")
            else:
                print(f"Warning: Patch folder not found: {wsi_patch_folder} for slide_id {base_slide_id} (from CSV: {slide_id_from_csv}). Skipping.")
        
        if not self.slide_data:
            raise RuntimeError(f"No valid WSI patch folders found. Please check paths, CSV content, and file existence.")
        print(f"Found {len(self.slide_data)} WSIs with patch paths for the dataset.")


    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        item_data = self.slide_data[idx]
        # Return paths and label, actual image loading and transformation happens in the training loop
        return {
            'patch_paths': item_data['patch_paths'], 
            'label': torch.tensor(item_data['label'], dtype=torch.long), # Or float for regression
            'slide_id': item_data['slide_id']
        }
    
    @staticmethod
    def patch_transforms(model_input_size=384, is_train=True):
        """
        Provides a basic set of image transformations.
        Call this in your training script to get the transform object.
        """
        mean = IMAGENET_DEFAULT_MEAN 
        std = IMAGENET_DEFAULT_STD

        class ToPILImage:
            def __call__(self, img):
                if isinstance(img, np.ndarray):
                    from PIL import Image
                    return Image.fromarray(img.astype('uint8'))
                return img
        
        if is_train:
            return transforms.Compose([
                transforms.Resize(model_input_size), 
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1
                    )
                ], p=0.8),
                # Example of RandStainNA integration:
                # RandStainNA(
                #     yaml_file=r"path/to/your/RandStainNA/config.yaml",
                #     std_hyper=-0.3,
                #     probability=1.0,
                #     distribution="normal",
                #     is_train=True,
                # ),
                # ToPILImage(), # Ensure PIL image before ToTensor if RandStainNA outputs numpy
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else: # Validation/Test
            return transforms.Compose([
                transforms.Resize(model_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
class SinglePatchDataset(Dataset):
    def __init__(self, patch_paths: List[str], transform: Callable):
        """
        Dataset for loading individual patches from a list of paths.

        Args:
            patch_paths (List[str]): A list of full file paths to the patch images.
            transform (Callable): The transformation function to apply to each patch.
        """
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        try:
            img = Image.open(patch_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading patch {patch_path}: {e}")
            return self.transform(Image.new('RGB', (224, 224))) # Return a dummy if error, or handle better