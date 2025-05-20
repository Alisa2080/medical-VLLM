from dataset.WSIBagDatasetMTL import WSIBagDatasetMIL
import torch
# Example of how to use the Dataset and DataLoader:
if __name__ == '__main__':

    # Create dataset instance
    try:
        dataset = WSIBagDatasetMIL(
            slide_list_csv=r"F:\dataset\CLAM-WSI\tumor_segment\process_list_autogen.csv",
            h5_coord_dir=r"F:\dataset\CLAM-WSI\tumor_segment\patches",
            wsi_dir=r"F:\dataset\CLAM-WSI\tumor",
            slide_id_column="slide_id", # Matching the dummy CSV
            label_column="label",
            model_input_size=384           # Matching the dummy CSV
        )
        print(f"Dataset created with {len(dataset)} slides.")

        if len(dataset) > 0:
            # Test __getitem__ for the first valid slide
            item0 = dataset[0]
            print(f"\nData for slide '{item0['slide_id']}':")
            print(f"  Label: {item0['label']}")
            print(f"  Number of patches: {item0['patches'].shape[0] if item0['patches'].numel() > 0 else 0}")
            if item0['patches'].numel() > 0:
                print(f"  Patches tensor shape: {item0['patches'].shape}") # (num_patches, C, H, W)
            print(f"  Error flag: {item0['error']}")


        # DataLoader example
        from torch.utils.data import DataLoader

        def mil_collate_fn(batch_list):
            # Filter out items that had errors or no patches
            # batch_list is a list of dicts from __getitem__
            valid_items = [item for item in batch_list if not item.get('error', False) and item['patches'].numel() > 0]
            if not valid_items:
                return None # Training loop should handle this
            return torch.utils.data.dataloader.default_collate(valid_items)

        dataloader = DataLoader(
            dataset,
            batch_size=1, # As requested, each item from loader is one WSI bag
            shuffle=False, # No shuffle for predictable test
            num_workers=0, # For simplicity in example
            collate_fn=mil_collate_fn
        )

        print("\nIterating through DataLoader (batch_size=1):")
        for i, batch_data in enumerate(dataloader):
            if batch_data is None:
                print(f"Batch {i+1} was skipped due to errors or no patches in source slide(s).")
                continue
            
            # batch_data is a dict where values are tensors batched along dim 0.
            # e.g., batch_data['patches'] is (1, num_patches, C, H, W) because batch_size=1
            slide_id = batch_data['slide_id'][0] # It's a list of one string
            patches = batch_data['patches'].squeeze(0) # Remove batch dim: (num_patches, C, H, W)
            label = batch_data['label'].squeeze(0)     # Remove batch dim: (scalar)
            
            print(f"  Batch {i+1}: Slide ID: {slide_id}, Label: {label.item()}, Num Patches: {patches.shape[0]}, Patches Shape: {patches.shape}")
            # In a real loop: model_output = model(patches)
            
    except Exception as e:
        print(f"An error occurred during dataset example: {e}")