import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split):
    # --- 修改代码：使用 os.path.basename 提取文件名 ---
    name = os.path.basename(path)
    # --- 修改代码结束 ---
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root, max_samples=None): # 添加 max_samples 参数
    with open(f"{root}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/train2014/*.jpg")) + list(glob(f"{root}/val2014/*.jpg"))
    random.shuffle(paths)
    # --- 修改代码：使用 os.path.basename 进行检查 ---
    caption_paths = [path for path in paths if os.path.basename(path) in iid2captions]
    # --- 修改代码结束 ---


    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        f"Total images: {len(paths)}, Images with captions: {len(caption_paths)}, Unique images in annotations: {len(iid2captions)}"
    )

    # --- 新增代码：限制样本数量 ---
    if max_samples is not None and max_samples > 0 and max_samples < len(caption_paths):
        caption_paths = caption_paths[:max_samples]
        print(f"Limiting to {len(caption_paths)} samples.")
        output_suffix = "_small" # 为小型数据集添加后缀
    else:
        print(f"Processing all {len(caption_paths)} samples.")
        output_suffix = "" # 完整数据集使用空后缀
    # --- 新增代码结束 ---


    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]
        if not batches: # 如果某个划分没有样本，则跳过
            print(f"No samples found for split: {split}. Skipping.")
            continue

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        # --- 修改代码：更新输出文件名 ---
        output_filename = f"{dataset_root}/coco_caption_karpathy{output_suffix}_{split}.arrow"
        print(f"Writing {len(batches)} samples for split '{split}' to {output_filename}")
        with pa.OSFile(output_filename, "wb") as sink:
        # --- 修改代码结束 ---
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

# --- 如何调用以生成小型数据集 (示例) ---
if __name__ == "__main__":
    coco_root_dir = r"F:\dataset\COCO" # 替换为你的 COCO 数据集根目录
    output_arrow_dir = r"E:\BaiduNetdiskDownload" # 替换为你的输出目录
    make_arrow(coco_root_dir, output_arrow_dir, max_samples=2000)
