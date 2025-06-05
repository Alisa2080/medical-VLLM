import os
from glob import glob
from .base_dataset import BaseDataset

class PMCDataset(BaseDataset):
    def __init__(self, data_dir, image_augmentation, split="train", image_size=384, 
                 max_text_len=196, draw_false_image=0, draw_false_text=0, 
                 image_only=False, source_name="pmc_json", **kwargs):
        """
        初始化 PMCDataset 类的实例。

        参数:
            data_dir (str): 包含 .arrow 文件的目录路径
            image_augmentation: 图像增强器
            split (str): 数据集的划分类型，可选值为 "train", "val", "test"
            image_size (int): 图像尺寸
            max_text_len (int): 最大文本长度
            draw_false_image (int): 随机抽取错误图像的数量
            draw_false_text (int): 随机抽取错误文本的数量
            image_only (bool): 是否仅使用图像
            source_name (str): 数据源名称，用于匹配文件名，默认为 "pmc_json"
            **kwargs: 其他传递给父类的参数
        """
        assert split in ["train", "val", "test"]
        
        # 对于测试集，通常使用验证集数据
        if split == "test":
            effective_split = "val"
        else:
            effective_split = split

        print(f"PMCDataset: Looking for files in directory: '{data_dir}'")
        print(f"PMCDataset: Split='{split}', effective_split='{effective_split}'")

        # 动态查找匹配的arrow文件
        # 文件名格式例如: pmc_json_train_0.arrow, pmc_json_val_0.arrow
        file_pattern = f"{source_name}_{effective_split}_*.arrow"
        arrow_files = glob(os.path.join(data_dir, file_pattern))
        
        # 从完整路径中提取文件名（不含扩展名）
        names = sorted([os.path.basename(f).replace(".arrow", "") for f in arrow_files])

        if not names:
            print(f"警告: 在目录 '{data_dir}' 中未找到与模式 '{file_pattern}' 匹配的 Arrow 文件。")
            print(f"目录内容: {os.listdir(data_dir) if os.path.exists(data_dir) else '目录不存在'}")
            # 可以选择抛出错误或允许空数据集
            # raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到与模式 '{file_pattern}' 匹配的 Arrow 文件。")

        print(f"为 PMC 数据集（split='{split}', effective_split='{effective_split}'）找到的 Arrow 文件名: {names}")
        
        # 调用父类的构造函数，传入所有必要的参数
        super().__init__(
            data_dir=data_dir,
            image_augmentation=image_augmentation,
            image_size=image_size,
            names=names,
            text_column_name="caption",  # PMC数据集的文本列名
            max_text_len=max_text_len,
            draw_false_image=draw_false_image,
            draw_false_text=draw_false_text,
            image_only=image_only,
            **kwargs  # 传递其他可能的参数
        )

    def __getitem__(self, index):
        # 对于纯文本训练，我们只需要文本及其相关信息
        return self.get_text_suite(index)