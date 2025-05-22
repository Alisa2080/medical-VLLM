import os
import shutil
import torch
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import unittest # Using unittest for a more structured approach

# Adjust the import path if your WSIBagDatasetMIL class is located elsewhere
from dataset.WSIBagDatasetMTL import WSIBagDatasetMIL

# --- Constants for dummy data ---
DUMMY_ROOT_DIR = "temp_test_WSIBagDatasetMIL"
DUMMY_CSV_FILENAME = "dummy_slide_list.csv"
DUMMY_PATCHES_SUBDIR = "patches_data"

class TestWSIBagDatasetMIL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up dummy data for all tests."""
        cls.dummy_root = DUMMY_ROOT_DIR
        cls.dummy_patches_root = os.path.join(cls.dummy_root, DUMMY_PATCHES_SUBDIR)
        cls.dummy_csv_path = os.path.join(cls.dummy_root, DUMMY_CSV_FILENAME)

        if os.path.exists(cls.dummy_root):
            shutil.rmtree(cls.dummy_root)
        os.makedirs(cls.dummy_root, exist_ok=True)
        os.makedirs(cls.dummy_patches_root, exist_ok=True)

        # 1. Create dummy CSV
        data = {
            'slide_id': ['slide01.svs', 'slide02.ndpi', 'slide03_no_folder.svs', 'slide04_empty_folder.svs', 'slide05.svs'],
            'label': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.dummy_csv_path, index=False)

        # 2. Create dummy patch files
        # Slide01
        slide01_dir = os.path.join(cls.dummy_patches_root, "slide01")
        os.makedirs(slide01_dir, exist_ok=True)
        cls._create_dummy_png(os.path.join(slide01_dir, "patch_0_0.png"))
        cls._create_dummy_png(os.path.join(slide01_dir, "patch_0_256.png"))
        cls._create_dummy_png(os.path.join(slide01_dir, "patch_256_0.png"))

        # Slide02
        slide02_dir = os.path.join(cls.dummy_patches_root, "slide02")
        os.makedirs(slide02_dir, exist_ok=True)
        cls._create_dummy_png(os.path.join(slide02_dir, "tile_x10_y20.jpg")) # Different extension
        cls._create_dummy_png(os.path.join(slide02_dir, "tile_x10_y276.jpeg"))

        # slide03_no_folder.svs - folder intentionally not created

        # slide04_empty_folder.svs - folder created but will be empty
        slide04_dir = os.path.join(cls.dummy_patches_root, "slide04_empty_folder")
        os.makedirs(slide04_dir, exist_ok=True)
        
        # Slide05 (will have no patches with .tiff extension if not specified in dataset)
        slide05_dir = os.path.join(cls.dummy_patches_root, "slide05")
        os.makedirs(slide05_dir, exist_ok=True)
        cls._create_dummy_png(os.path.join(slide05_dir, "img_01.tiff"))


        print(f"Dummy CSV created at: {cls.dummy_csv_path}")
        print(f"Dummy patches root created at: {cls.dummy_patches_root}")

    @classmethod
    def tearDownClass(cls):
        """Clean up dummy data after all tests."""
        if os.path.exists(cls.dummy_root):
            shutil.rmtree(cls.dummy_root)
        print("Dummy data cleaned up.")

    @staticmethod
    def _create_dummy_png(path, size=(32, 32)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(array)
        img.save(path)

    def test_01_dataset_initialization(self):
        print("\n--- Testing WSIBagDatasetMIL Initialization ---")
        dataset = WSIBagDatasetMIL(
            slide_list_csv=self.dummy_csv_path,
            patches_root_dir=self.dummy_patches_root,
            slide_id_column="slide_id",
            label_column="label",
            model_input_size=224, # This is for patch_transforms, not directly used by __getitem__
            patch_file_extensions=('.png', '.jpg', '.jpeg') # Exclude .tiff for slide05 test
        )
        # Expected: slide01, slide02.
        # slide03 (no folder), slide04 (empty folder), slide05 (only tiff) should be skipped/warned.
        self.assertEqual(len(dataset), 2, "Dataset should find 2 valid WSIs (slide01, slide02).")
        print(f"Dataset initialized. Number of WSIs found: {len(dataset)}")
        return dataset # Pass dataset to other tests

    def test_02_getitem_functionality(self):
        print("\n--- Testing __getitem__ ---")
        dataset = self.test_01_dataset_initialization() # Get dataset from previous test
        if not dataset or len(dataset) == 0:
            self.skipTest("Dataset is empty, skipping __getitem__ test.")

        # Test first item (should be slide01)
        item0 = dataset[0]
        self.assertEqual(item0['slide_id'], "slide01")
        self.assertIsInstance(item0['label'], torch.Tensor)
        self.assertEqual(item0['label'].item(), 0)
        self.assertIsInstance(item0['patch_paths'], list)
        self.assertEqual(len(item0['patch_paths']), 3) # 3 PNGs for slide01
        if item0['patch_paths']:
            self.assertTrue(item0['patch_paths'][0].endswith(".png"))
        print(f"Item 0 (Slide ID: {item0['slide_id']}): Label: {item0['label']}, Num Patch Paths: {len(item0['patch_paths'])}")

        # Test second item (should be slide02)
        if len(dataset) > 1:
            item1 = dataset[1]
            self.assertEqual(item1['slide_id'], "slide02")
            self.assertEqual(item1['label'].item(), 1)
            self.assertIsInstance(item1['patch_paths'], list)
            self.assertEqual(len(item1['patch_paths']), 2) # 2 JPG/JPEG for slide02
            if item1['patch_paths']:
                 self.assertTrue(item1['patch_paths'][0].endswith(".jpg") or item1['patch_paths'][0].endswith(".jpeg"))
            print(f"Item 1 (Slide ID: {item1['slide_id']}): Label: {item1['label']}, Num Patch Paths: {len(item1['patch_paths'])}")


    def test_03_patch_transforms_static_method(self):
        print("\n--- Testing patch_transforms static method ---")
        model_size = 256
        train_transforms = WSIBagDatasetMIL.patch_transforms(model_input_size=model_size, is_train=True)
        eval_transforms = WSIBagDatasetMIL.patch_transforms(model_input_size=model_size, is_train=False)
        
        self.assertIsNotNone(train_transforms)
        self.assertIsNotNone(eval_transforms)
        print(f"Train transforms created: {type(train_transforms)}")
        print(f"Eval transforms created: {type(eval_transforms)}")

        # Test applying the transform to a dummy image
        dummy_pil_img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        transformed_tensor = train_transforms(dummy_pil_img)
        self.assertIsInstance(transformed_tensor, torch.Tensor)
        self.assertEqual(transformed_tensor.shape, (3, model_size, model_size))
        print(f"Dummy image transformed shape: {transformed_tensor.shape}")


    def test_04_dataloader_and_patch_simulation(self):
        print("\n--- Testing DataLoader and Patch Loading Simulation ---")
        dataset = self.test_01_dataset_initialization()
        if not dataset or len(dataset) == 0:
            self.skipTest("Dataset is empty, skipping DataLoader test.")

        # Using default collate_fn as __getitem__ returns simple structures
        # batch_size=1 is typical for WSI-level processing before patch batching
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        self.assertTrue(len(dataloader) == len(dataset), "DataLoader length should match dataset length.")

        print(f"Iterating through DataLoader (dataloader batch_size=1, dataset size={len(dataset)}):")
        
        # Get transforms for simulation
        test_transforms = WSIBagDatasetMIL.patch_transforms(model_input_size=dataset.model_input_size, is_train=False)

        for i, batch_data in enumerate(dataloader):
            self.assertIn('slide_id', batch_data)
            self.assertIn('patch_paths', batch_data)
            self.assertIn('label', batch_data)

            # batch_data['slide_id'] will be a list of one string because batch_size=1
            # batch_data['patch_paths'] will be a list containing one list of paths
            slide_id = batch_data['slide_id'][0] 
            patch_paths_for_current_wsi = batch_data['patch_paths'][0] 
            label = batch_data['label'] # Tensor of shape [1]

            print(f"\n  DataLoader Batch {i+1}: Slide ID: {slide_id}, Label: {label.item()}, Num Patch Paths: {len(patch_paths_for_current_wsi)}")
            self.assertTrue(len(patch_paths_for_current_wsi) > 0, f"Slide {slide_id} should have patch paths.")

            # Simulate loading and transforming a few patches for this WSI
            print(f"    Simulating loading & transforming first patch for {slide_id}:")
            
            if patch_paths_for_current_wsi:
                p_path = patch_paths_for_current_wsi[0]
                try:
                    img = Image.open(p_path).convert('RGB')
                    transformed_patch = test_transforms(img)
                    print(f"      Loaded and transformed patch: {os.path.basename(p_path)}, Shape: {transformed_patch.shape}")
                    self.assertEqual(transformed_patch.shape, (3, dataset.model_input_size, dataset.model_input_size))
                except Exception as e_patch:
                    self.fail(f"Error loading/transforming patch {p_path}: {e_patch}")
            else:
                 self.fail(f"Patch path list for slide {slide_id} is empty in DataLoader item.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)