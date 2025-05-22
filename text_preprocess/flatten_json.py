import json
import os

# --- 配置区域 ---
# 这是包含 List[List[str]] 结构JSON文件的目录
# (即您上一个脚本 "OUTPUT_CLEANED_JSON_DIR" 的输出目录)
INPUT_NESTED_JSON_DIR = r'F:\dataset\medical_text_json\version3' # 或者您存放 v3 版本文件的目录

# 这是您希望保存扁平化后 List[str] 结构JSON文件的新目录
OUTPUT_FLATTENED_JSON_DIR = r'F:\dataset\medical_text_json\version4'

def flatten_nested_json_lists(input_dir: str, output_dir: str):
    """
    读取输入目录中所有包含 List[List[str]] 结构的JSON文件，
    将其内容扁平化为 List[str]，并保存到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    json_files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    total_files = len(json_files_to_process)
    print(f"Found {total_files} JSON files in '{input_dir}' to flatten.")

    processed_count = 0
    error_count = 0

    for i, filename in enumerate(json_files_to_process):
        input_file_path = os.path.join(input_dir, filename)
        # 为输出文件名添加一个后缀，以区分
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_flattened{ext}"
        output_file_path = os.path.join(output_dir, output_filename)

        print(f"Processing file {i+1}/{total_files}: {filename}...")

        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                nested_list_data = json.load(f)

            if not isinstance(nested_list_data, list):
                print(f"  Warning: Content of {filename} is not a list. Skipping.")
                error_count += 1
                continue

            flattened_list = []
            for inner_list in nested_list_data:
                if isinstance(inner_list, list):
                    for item in inner_list:
                        if isinstance(item, str):
                            flattened_list.append(item)
                        else:
                            print(f"  Warning: Item '{str(item)[:30]}...' in an inner list of {filename} is not a string. Skipping this item.")
                else:
                    # 如果外层列表的元素不是列表而是字符串（兼容之前可能的单层列表结构）
                    if isinstance(inner_list, str):
                        flattened_list.append(inner_list)
                    else:
                        print(f"  Warning: Item '{str(inner_list)[:30]}...' in the outer list of {filename} is not a list or a string. Skipping this item.")
            
            if flattened_list:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(flattened_list, f, ensure_ascii=False, indent=2)
                # print(f"  Successfully flattened and saved to {output_file_path} ({len(flattened_list)} segments)")
                processed_count +=1
            else:
                print(f"  No segments to save after flattening for {filename}.")
                # 即使是空列表也算处理过，但不计入成功保存有效内容的文件
                # error_count +=1 # 或者根据需求决定是否算作错误

        except json.JSONDecodeError as jde:
            print(f"  Error decoding JSON from {filename}: {jde}. Skipping.")
            error_count += 1
        except Exception as e:
            print(f"  An unexpected error occurred while processing {filename}: {e}. Skipping.")
            error_count += 1
            
    print(f"\n--- Flattening Summary ---")
    print(f"Total files found: {total_files}")
    print(f"Files successfully processed and saved: {processed_count}")
    print(f"Files skipped or with errors: {error_count}")


if __name__ == '__main__':
    # 确保输入目录存在
    if not os.path.isdir(INPUT_NESTED_JSON_DIR):
        print(f"错误：输入目录 '{INPUT_NESTED_JSON_DIR}' 未找到。请确保路径正确。")
    else:
        print(f"--- Starting JSON List Flattening Process ---")
        print(f"Input directory (nested lists): {INPUT_NESTED_JSON_DIR}")
        print(f"Output directory (flattened lists): {OUTPUT_FLATTENED_JSON_DIR}")
        
        flatten_nested_json_lists(INPUT_NESTED_JSON_DIR, OUTPUT_FLATTENED_JSON_DIR)
        
        print(f"--- JSON List Flattening Finished ---")