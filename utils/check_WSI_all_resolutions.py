import os
import openslide
import pandas as pd
import argparse

def get_wsi_level_dimensions(wsi_path):
    """
    Opens a WSI file and returns its available level dimensions.
    """
    try:
        wsi = openslide.open_slide(wsi_path)
        level_dims = wsi.level_dimensions
        wsi.close()
        return level_dims
    except openslide.OpenSlideError as e:
        print(f"Error opening or reading {wsi_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with {wsi_path}: {e}")
        return None

def main(wsi_folder_path, output_csv_path):
    """
    Processes all WSI files in a folder and saves their level dimensions to a CSV.
    """
    print(f"Attempting to walk through directory: {wsi_folder_path}") # 新增打印
    if not os.path.isdir(wsi_folder_path): # 新增检查
        print(f"Error: The provided path '{wsi_folder_path}' is not a valid directory or does not exist.")
        return
    wsi_files = []
    # You can add more WSI file extensions if needed
    found_any_file = False # 新增标志
    supported_extensions = ('.svs', '.tif', '.tiff', '.vms', '.vmu', '.ndpi', 
                            '.scn', '.mrxs', '.bif', '.zif') 
    for root, _, files in os.walk(wsi_folder_path):
        print(f"Checking directory: {root}") # 新增打印
        found_any_file = False # 新增标志
        for file in files:
            if file.lower().endswith(supported_extensions):
                wsi_files.append(os.path.join(root, file))

    if not found_any_file: # 新增检查
        print(f"Warning: os.walk did not seem to iterate through any files/folders in '{wsi_folder_path}'. Check path and permissions.")
    if not wsi_files:
        print(f"No WSI files found in {wsi_folder_path} with supported extensions.")
        return

    print(f"Found {len(wsi_files)} WSI files to process.")

    results = []
    for wsi_path in wsi_files:
        filename = os.path.basename(wsi_path)
        print(f"Processing: {filename}")
        level_dimensions = get_wsi_level_dimensions(wsi_path)
        if level_dimensions:
            results.append({
                'filename': filename,
                'full_path': wsi_path,
                'level_dimensions': str(level_dimensions) # Store as string for CSV
            })
        else:
            results.append({
                'filename': filename,
                'full_path': wsi_path,
                'level_dimensions': 'Error reading dimensions'
            })

    df = pd.DataFrame(results)
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved WSI level dimensions to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file to {output_csv_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WSI level dimensions and save to CSV.")
    parser.add_argument('--wsi_folder', type=str, default=r"F:\dataset\CLAM-WSI\tumor",
                        help="Path to the folder containing WSI files.")
    parser.add_argument('--output_csv', type=str,default=r"F:\dataset\CLAM-WSI\tumor_segment\wsi_dimensions.csv",
                        help="Path to save the output CSV file.")
    args = parser.parse_args()
    
    main(args.wsi_folder, args.output_csv)