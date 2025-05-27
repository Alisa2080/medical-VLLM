# Vision_Encoder/scripts/batch_assess_patches.py
import os
import pandas as pd
import argparse
from wsi_core.patch_quality_assessor import assess_wsi_patches, PatchQualityAssessor
from multiprocessing import Pool
import functools

def process_single_wsi(wsi_folder, patches_root_dir, output_dir, assessor):
    """处理单个WSI的patch质量评估"""
    wsi_patch_folder = os.path.join(patches_root_dir, wsi_folder)
    if not os.path.isdir(wsi_patch_folder):
        return None
    
    output_file = os.path.join(output_dir, f"{wsi_folder}_quality.csv")
    
    # 如果已经存在，跳过
    if os.path.exists(output_file):
        print(f"Quality file already exists for {wsi_folder}, skipping...")
        return output_file
    
    try:
        df = assess_wsi_patches(wsi_patch_folder, output_file, assessor)
        print(f"Completed assessment for {wsi_folder}: {len(df)} patches")
        return output_file
    except Exception as e:
        print(f"Error processing {wsi_folder}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch assess patch quality')
    parser.add_argument('--patches_root_dir', type=str,default=r"F:\dataset\CLAM-WSI\tumor_segment\patches",
                        help='Root directory containing WSI patch folders')
    parser.add_argument('--output_dir', type=str, default=r"F:\dataset\CLAM-WSI\tumor_segment\patch_score",
                        help='Directory to save quality assessment results')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of parallel workers')
    parser.add_argument('--wsi_list', type=str, default=None,
                        help='Optional CSV file with list of WSI IDs to process')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取WSI列表
    if args.wsi_list:
        wsi_df = pd.read_csv(args.wsi_list)
        # 从slide_id中提取基础名称（去掉扩展名）
        wsi_folders = [os.path.splitext(str(sid))[0] for sid in wsi_df['slide_id'].tolist()]
    else:
        wsi_folders = [f for f in os.listdir(args.patches_root_dir) 
                      if os.path.isdir(os.path.join(args.patches_root_dir, f))]
    
    print(f"Found {len(wsi_folders)} WSI folders to process")
    
    # 创建质量评估器
    assessor = PatchQualityAssessor()
    
    # 并行处理
    process_func = functools.partial(
        process_single_wsi,
        patches_root_dir=args.patches_root_dir,
        output_dir=args.output_dir,
        assessor=assessor
    )
    
    with Pool(args.num_workers) as pool:
        results = pool.map(process_func, wsi_folders)
    
    # 合并所有结果
    all_dfs = []
    for result_file in results:
        if result_file and os.path.exists(result_file):
            df = pd.read_csv(result_file)
            all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_output = os.path.join(args.output_dir, 'all_patches_quality.csv')
        combined_df.to_csv(combined_output, index=False)
        print(f"Combined results saved to {combined_output}")
        print(f"Total patches assessed: {len(combined_df)}")
    
    print("Batch assessment completed!")

if __name__ == '__main__':
    main()