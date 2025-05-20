import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import openslide
import h5py
from torchvision import transforms # For the example transform
import numpy as np
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from RandStainNA.randstainna import RandStainNA

class ToPILImage:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            from PIL import Image
            return Image.fromarray(img.astype('uint8'))
        return img

class WSIBagDatasetMIL(Dataset):
    def __init__(self,
                 slide_list_csv: str,
                 h5_coord_dir: str,
                 wsi_dir: str,
                 label_column: str = "label",
                 slide_id_column: str = "slide_id",
                 is_train: bool = True,
                 model_input_size: int = 384
                 ):
        self.h5_coord_dir = h5_coord_dir
        self.wsi_dir = wsi_dir
        self.img_transforms = self.patch_transforms(model_input_size=model_input_size, is_train=is_train) 
        self.label_column = label_column
        self.slide_id_column = slide_id_column
        self.model_input_size = model_input_size

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
            slide_id_with_ext = str(row[self.slide_id_column]) # Ensure slide_id is a string
            label = row[self.label_column]
            base_slide_id = slide_id_with_ext
            possible_extensions = [".svs", ".tif", ".tiff", ".vsi", ".ndpi"] # Add other relevant WSI extensions
            for ext_to_check in possible_extensions:
                if slide_id_with_ext.lower().endswith(ext_to_check.lower()):
                    base_slide_id = slide_id_with_ext[:-len(ext_to_check)]
                    break
            h5_path = os.path.join(self.h5_coord_dir, f"{base_slide_id}.h5")
            wsi_path = os.path.join(self.wsi_dir, f"{base_slide_id}.svs") # Use the original self.slide_ext for WSI
            if os.path.exists(h5_path) and os.path.exists(wsi_path):
                self.slide_data.append({'slide_id': base_slide_id, 'label': label, 'h5_path': h5_path, 'wsi_path': wsi_path})
            else:
                print(f"Warning: Skipping slide '{base_slide_id}'. File(s) not found.")
                if not os.path.exists(h5_path):
                    print(f"  Missing HDF5: {h5_path}")
                if not os.path.exists(wsi_path):
                    print(f"  Missing WSI: {wsi_path}")
        
        if not self.slide_data:
            raise RuntimeError(f"No valid slides found after checking files listed in {slide_list_csv}. Please check paths and file existence.")
        print(f"Found {len(self.slide_data)} valid slides for the dataset.")


    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        item_data = self.slide_data[idx]
        slide_id = item_data['slide_id']
        label = item_data['label']
        h5_path = item_data['h5_path']
        wsi_path = item_data['wsi_path']
        wsi = None # Initialize wsi to None for finally block

        try:
            wsi = openslide.open_slide(wsi_path)
            with h5py.File(h5_path, 'r') as h5_file:
                if 'coords' not in h5_file:
                    raise ValueError(f"'coords' dataset not found in HDF5 file: {h5_path}")
                patch_coords = h5_file['coords'][:]
                
                # Assuming create_patches_fp.py (CLAM version) saves patch_size 
                # as an attribute of the 'coords' dataset, and these coordinates are at level 0.
                # The patch_size attribute then refers to the size of the patch at level 0.
                if 'patch_size' not in h5_file['coords'].attrs:
                    raise ValueError(f"'patch_size' attribute not found in 'coords' dataset in HDF5 file: {h5_path}")
                patch_size_from_h5 = h5_file['coords'].attrs['patch_size']
                # patch_level_from_h5 = h5_file['coords'].attrs.get('patch_level', 0) # Default to 0 if not present

                all_patches_tensors = []
                for coord_x, coord_y in patch_coords:
                    try:
                        # Read region at level 0 using the patch_size from H5 attributes.
                        # This assumes coordinates are for level 0 and patch_size_from_h5 is the size at level 0.
                        patch_img_pil = wsi.read_region(
                            (int(coord_x), int(coord_y)), # Coordinates must be int
                            0, # Assuming level 0, consistent with CLAM's extract_features_fp.py
                            (int(patch_size_from_h5), int(patch_size_from_h5))
                        ).convert('RGB')

                        transformed_patch = self.img_transforms(patch_img_pil)
                        all_patches_tensors.append(transformed_patch)
                    except Exception as e_patch:
                        print(f"Warning: Error reading or transforming patch at ({coord_x}, {coord_y}) for slide {slide_id}: {e_patch}. Skipping patch.")
                        continue 

                if not all_patches_tensors:
                    print(f"Warning: No valid patches could be loaded for slide {slide_id} from {h5_path}. This slide will have 0 patches.")
                    patches_tensor = torch.empty(0) 
                else:
                    patches_tensor = torch.stack(all_patches_tensors, dim=0)

        except Exception as e:
            print(f"Error processing slide {slide_id} (WSI: {wsi_path}, H5: {h5_path}): {e}")
            return { # Return a structure that collate_fn can identify as erroneous
                'patches': torch.empty(0), 
                'label': torch.tensor(-1, dtype=torch.long), 
                'slide_id': slide_id,
                'error': True # Flag to indicate an error
            }
        finally:
            if wsi:
                wsi.close()
        
        return {
            'patches': patches_tensor, 
            'label': torch.tensor(label, dtype=torch.long),
            'slide_id': slide_id,
            'error': False
        }

# Example transform function (adapt this to match your ViT pre-training)
    def patch_transforms(self,model_input_size=384, is_train=True):
        """
        Provides a basic set of image transformations for MIL fine-tuning.
        Ensure these are consistent with your ViT's pre-training, especially normalization.
        """
        mean = IMAGENET_DEFAULT_MEAN 
        std = IMAGENET_DEFAULT_STD

        if is_train:
             # Added transforms.Resize and return statement
            return transforms.Compose([
                transforms.Resize(model_input_size), # Ensure patches are resized
                transforms.RandomHorizontalFlip(p=0.5), # Added RandomHorizontalFlip
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
                # # RandStainNA( # This expects a PIL image as input
                #     yaml_file=r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml", # Corrected path for example
                #     std_hyper=-0.3,
                #     probability=1.0,
                #     distribution="normal",
                #     is_train=True,
                # ),
                # ToPILImage(), # RandStainNA might return numpy, ensure it's PIL before ToTensor
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
  
        else: # Validation/Test
            return transforms.Compose([
            transforms.Resize(model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

