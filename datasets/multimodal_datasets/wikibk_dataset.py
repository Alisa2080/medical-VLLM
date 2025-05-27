from glob import glob
from .base_dataset import BaseDataset


class WikibkDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        """
        初始化 WikibkDataset 类的实例。

        参数:
            *args: 传递给父类构造函数的位置参数。
            split (str): 数据集的划分类型，可选值为 "train", "val", "test"。
            **kwargs: 传递给父类构造函数的关键字参数。
        """
        # 确保传入的 split 参数是有效的划分类型
        assert split in ["train", "val", "test"]
        # 由于测试集使用验证集数据，将 "test" 替换为 "val"
        if split == "test":
            split = "val"

        # 根据不同的划分类型，设置对应的数据集名称列表
        if split == "train":
            # 训练集包含 50 个文件
            names = [f"wikibk_train_{i}" for i in range(1)]
        elif split == "val":
            # 验证集只有一个文件
            names = ["wikibk_val_0"]

        # 调用父类的构造函数，传入必要的参数
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_text_suite(index)
