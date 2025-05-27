import os
from glob import glob
from .base_dataset import BaseDataset

class PMCDataset(BaseDataset):
    def __init__(self, *args, data_dir="", split="", source_name="pmc_json", **kwargs):
        """
        初始化 PMCDataset 类的实例。

        参数:
            *args: 传递给父类构造函数的位置参数。
            data_dir (str): 包含 .arrow 文件的目录。
            split (str): 数据集的划分类型，可选值为 "train", "val", "test"。
            source_name (str): 数据源名称，用于匹配文件名，默认为 "pmc_json"。
            **kwargs: 传递给父类构造函数的关键字参数。
        """
        assert split in ["train", "val", "test"]
        if split == "test": # 通常测试集使用验证集数据
            effective_split = "val"
        else:
            effective_split = split

        # 动态查找匹配的arrow文件
        # 文件名格式例如: pmc_json_train_0.arrow, pmc_json_val_0.arrow
        file_pattern = f"{source_name}_{effective_split}_*.arrow"
        arrow_files = glob(os.path.join(data_dir, file_pattern))
        
        names = sorted([os.path.basename(f).replace(".arrow", "") for f in arrow_files])

        if not names:
            print(f"警告: 在目录 '{data_dir}' 中未找到与模式 '{file_pattern}' 匹配的 Arrow 文件。")
            # 可以选择抛出错误或允许空数据集
            # raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到与模式 '{file_pattern}' 匹配的 Arrow 文件。")

        print(f"为 PMC 数据集（split='{split}', effective_split='{effective_split}'）找到的 Arrow 文件名: {names}")
        
        # 调用父类的构造函数，传入必要的参数
        # text_column_name 仍然是 "caption"，因为 path2rest 中是这样定义的
        super().__init__(*args, **kwargs, data_dir=data_dir, names=names, text_column_name="caption")

    def __getitem__(self, index):
        # 对于纯文本训练，我们只需要文本及其相关信息
        return self.get_text_suite(index)