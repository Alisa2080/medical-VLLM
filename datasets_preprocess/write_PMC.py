import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
from tqdm import tqdm
from glob import glob


def path2rest(text_content, source_name="pmc_json", split_type="train"):
    """
    将输入的文本内容（例如一个段落）转换为包含图像、文本、来源和划分信息的列表。

    参数:
        text_content (str): 输入的文本内容。
        source_name (str): 数据来源名称 (例如 "pmc_json", "wikibk")。
        split_type (str): 数据划分类型 (例如 "train", "val")。

    返回:
        list: 包含图像、文本、来源和划分信息的列表。
    """
    return [
        "None",  # 图像信息为空
        [text_content],  # 文本内容，以列表形式存储
        source_name,
        split_type,
    ]


def make_arrow(json_input_dir, dataset_root, mode, source_name="pmc_json", paragraphs_per_arrow_file=100000):
    """
    将指定目录下的 JSON 文件内容（每个文件包含一个段落列表）转换为 Apache Arrow 格式的数据文件。

    参数:
        json_input_dir (str): 包含 JSON 文件的目录路径。
        dataset_root (str): 用于存储生成的 Arrow 文件的目标目录路径。
        mode (str): "train" 或 "val"，指示当前处理的数据集划分。
        source_name (str): 数据集的名称，例如 "pmc_json"。
        paragraphs_per_arrow_file (int): 每个 Arrow 文件大约包含多少段落。
                                         设置为 -1 则表示该 mode 下的所有段落合并到一个 Arrow 文件中。
    """
    all_json_files = glob(os.path.join(json_input_dir, "*.json"))
    if not all_json_files:
        print(f"在目录 {json_input_dir} 中未找到 JSON 文件。")
        return

    # 随机打乱文件列表以进行训练/验证集划分
    random.shuffle(all_json_files)
    
    # 定义训练集/验证集的划分比例，例如80%训练集
    num_total_files = len(all_json_files)
    num_train_files = int(num_total_files * 0.8)

    if mode == "train":
        target_json_files = all_json_files[:num_train_files]
        print(f"正在为 TRAIN 集处理 {len(target_json_files)} 个 JSON 文件。")
    elif mode == "val":
        target_json_files = all_json_files[num_train_files:]
        print(f"正在为 VALIDATION 集处理 {len(target_json_files)} 个 JSON 文件。")
    else:
        print(f"无效的 mode: {mode}。请选择 'train' 或 'val'。")
        return

    if not target_json_files:
        print(f"没有文件可供处理，mode: {mode}")
        return

    os.makedirs(dataset_root, exist_ok=True)

    # 如果 paragraphs_per_arrow_file 为 -1, 设置为一个极大值，以便所有段落写入一个文件
    if paragraphs_per_arrow_file == -1:
        chunk_size = float('inf')
    else:
        chunk_size = paragraphs_per_arrow_file

    all_paragraphs_for_mode = []
    print(f"正在从JSON文件中读取 {mode} 集的段落...")
    for json_file_path in tqdm(target_json_files, desc=f"读取 {mode} JSONs"):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                # 假设每个JSON文件包含一个段落字符串的列表
                paragraphs_in_file = json.load(f)
                if isinstance(paragraphs_in_file, list):
                    for p_text in paragraphs_in_file:
                        if isinstance(p_text, str) and p_text.strip():
                            cleaned_text = p_text.strip()
                            if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                                cleaned_text = cleaned_text[1:-1]
                            all_paragraphs_for_mode.append(cleaned_text)
                        # else:
                        #     print(f"警告: 文件 {json_file_path} 中的一个元素不是字符串或为空，已跳过。内容: {p_text[:50]}...")
                else:
                    print(f"警告: 文件 {json_file_path} 的内容不是列表。已跳过。")
        except json.JSONDecodeError:
            print(f"错误: 解析文件 {json_file_path} 中的JSON时出错。已跳过。")
        except Exception as e:
            print(f"处理文件 {json_file_path} 时发生意外错误: {e}。已跳过。")

    if not all_paragraphs_for_mode:
        print(f"在 {mode} 模式下未提取到任何段落。没有内容可写入 Arrow 文件。")
        return
    
    print(f"总共为 {mode} 集提取了 {len(all_paragraphs_for_mode)} 个段落。")

    arrow_file_idx = 0
    # 确定分块大小
    if paragraphs_per_arrow_file == -1:
        # 如果列表不为空，则块大小为列表的总长度，以确保一次处理所有内容
        chunk_size = len(all_paragraphs_for_mode)
    else:
        chunk_size = paragraphs_per_arrow_file
        if chunk_size <= 0: # 确保 chunk_size 是正数
            print(f"警告: paragraphs_per_arrow_file ({paragraphs_per_arrow_file}) 无效，将使用默认值 1。")
            
            chunk_size = 1 # 或者一个合理的默认值，例如100000，但至少为1
    # 确保 chunk_size 至少为 1，以避免 range() 的 step 参数为0
    if chunk_size == 0 and len(all_paragraphs_for_mode) > 0:
        # 这种情况理论上不应该发生，因为如果 len > 0 且 paragraphs_per_arrow_file == -1, chunk_size 会是 len
        # 如果 paragraphs_per_arrow_file > 0 且被设为0，上面的 if chunk_size <= 0 会处理
        print(f"警告: 计算得到的 chunk_size 为0，但仍有段落需要处理。将 chunk_size 设置为列表长度。")
        chunk_size = len(all_paragraphs_for_mode)
    elif chunk_size == 0 and len(all_paragraphs_for_mode) == 0:
        # 如果列表为空，之前的检查已经 return，这里只是为了完整性
        pass

    for i in range(0, len(all_paragraphs_for_mode), chunk_size):
        current_chunk_paragraphs = all_paragraphs_for_mode[i : i + chunk_size]
        if not current_chunk_paragraphs:
            continue

        print(f"\n正在准备写入 {source_name}_{mode}_{arrow_file_idx}.arrow，包含 {len(current_chunk_paragraphs)} 个段落。")
        
        # 使用列表推导式，将每个段落转换为包含图像、文本、来源和划分信息的列表
        bs = [path2rest(p, source_name, mode) for p in tqdm(current_chunk_paragraphs, desc="转换为Arrow格式")]
        
        if not bs:
            print(f"没有数据可写入 {source_name}_{mode}_{arrow_file_idx}.arrow，已跳过。")
            arrow_file_idx += 1
            continue

        # 将处理后的列表转换为 Pandas 数据框，并指定列名
        dataframe = pd.DataFrame(bs, columns=["image", "caption", "source", "split"])

        # 将 Pandas 数据框转换为 Apache Arrow 表
        table = pa.Table.from_pandas(dataframe)

        output_arrow_file = f"{dataset_root}/{source_name}_{mode}_{arrow_file_idx}.arrow"
        # 以二进制写入模式打开一个新的 Arrow 文件
        with pa.OSFile(output_arrow_file, "wb") as sink:
            # 创建一个 Arrow 记录批处理文件写入器
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                # 将 Arrow 表写入文件
                writer.write_table(table)
        print(f"成功写入 {output_arrow_file}")

        # 删除不再使用的对象以释放内存
        del dataframe
        del table
        del bs
        gc.collect()
        arrow_file_idx += 1

if __name__ == "__main__":
    # 假设您的 JSON 文件存储在 "F:\dataset\medical_text_json\version4"
    # 请将其修改为您的实际JSON文件目录路径
    json_data_dir = r"F:\dataset\medical_text_json\version4" 
    
    # 假设您希望将生成的 Arrow 文件存储在 "e:\article_code\vlmo\data\pmc_json_arrow"
    # 请将其修改为您的目标 Arrow 文件输出目录路径
    arrow_output_dir = r"F:\dataset\Medical_TEXT"

    # 每个 Arrow 文件包含的段落数量。设置为 -1 则每个 mode（train/val）只生成一个 Arrow 文件。
    # 如果您的数据集非常大，建议设置一个合理的数值（例如 100000 或 500000）来分块。
    paragraphs_per_file = -1 

    print("开始处理训练数据...")
    make_arrow(json_input_dir=json_data_dir,
               dataset_root=arrow_output_dir,
               mode="train",
               source_name="pmc_json", # 您可以自定义来源名称
               paragraphs_per_arrow_file=paragraphs_per_file)
    
    print("\n开始处理验证数据...")
    make_arrow(json_input_dir=json_data_dir,
               dataset_root=arrow_output_dir,
               mode="val",
               source_name="pmc_json", # 与训练数据保持一致
               paragraphs_per_arrow_file=paragraphs_per_file)
    
    print("\n所有处理完成。")