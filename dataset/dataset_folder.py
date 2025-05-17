# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Modified on torchvision code bases
# https://github.com/pytorch/vision
# --------------------------------------------------------'
from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import random
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# 定义允许的图像扩展名
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """
    检查文件是否具有允许的扩展名。

    参数:
        filename (str): 文件的路径。
        extensions (Tuple[str, ...]): 要考虑的扩展名元组（小写）。

    返回:
        bool: 如果文件名以给定的扩展名之一结尾，则返回 True，否则返回 False。
    """
    # 将文件名转换为小写，并检查是否以 extensions 中的某个扩展名结尾
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """
    检查文件是否为允许的图像扩展名。

    参数:
        filename (string): 文件的路径

    返回:
        bool: 如果文件名以已知的图像扩展名结尾，则返回 True，否则返回 False
    """
    # 调用 has_file_allowed_extension 函数检查文件名是否以已知的图像扩展名结尾
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    # 数据集的根目录
    directory: str,
    # 类别名称到类别索引的映射字典
    class_to_idx: Dict[str, int],
    # 允许的文件扩展名元组，可选参数
    extensions: Optional[Tuple[str, ...]] = None,
    # 用于检查文件是否有效的函数，可选参数
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """
    从指定目录中创建数据集，将符合条件的文件及其对应的类别索引组合成元组列表。

    参数:
    directory (str): 数据集的根目录。
    class_to_idx (Dict[str, int]): 类别名称到类别索引的映射字典。
    extensions (Optional[Tuple[str, ...]]): 允许的文件扩展名元组，默认为 None。
    is_valid_file (Optional[Callable[[str], bool]]): 用于检查文件是否有效的函数，默认为 None。

    返回:
    List[Tuple[str, int]]: 包含文件路径和对应类别索引的元组列表。
    """
    # 用于存储有效的文件路径和对应类别索引的元组列表
    instances = []
    # 扩展目录路径中的用户主目录符号（如 ~）
    directory = os.path.expanduser(directory)
    # 检查 extensions 和 is_valid_file 是否同时为 None
    both_none = extensions is None and is_valid_file is None
    # 检查 extensions 和 is_valid_file 是否同时不为 None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        # 如果同时为 None 或同时不为 None，则抛出异常
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        # 如果 extensions 不为 None，定义一个内部函数用于检查文件是否具有允许的扩展名
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    # 确保 is_valid_file 是一个可调用的函数
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    # 遍历所有类别名称，并按字母顺序排序
    for target_class in sorted(class_to_idx.keys()):
        # 获取当前类别的类别索引
        class_index = class_to_idx[target_class]
        # 构建当前类别的目录路径
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            # 如果目录不存在，则跳过该类别
            continue
        # 递归遍历当前类别目录下的所有文件
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            # 遍历当前目录下的所有文件名，并按字母顺序排序
            for fname in sorted(fnames):
                # 构建当前文件的完整路径
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    # 如果文件有效，则将文件路径和类别索引组合成元组添加到 instances 列表中
                    item = path, class_index
                    instances.append(item)
    # 返回包含有效文件路径和对应类别索引的元组列表
    return instances
def default_loader(path: str) -> Any:
    """
    根据当前的图像后端选择合适的加载器来加载图像。

    参数:
        path (str): 图像文件的路径。

    返回:
        Any: 加载的图像对象，可能是accimage.Image或PIL.Image对象。
    """
    # 从torchvision中获取当前的图像后端
    from torchvision import get_image_backend
    # 检查当前的图像后端是否为accimage
    if get_image_backend() == 'accimage':
        # 如果是accimage，则使用accimage_loader加载图像
        return accimage_loader(path)
    else:
        # 否则，使用pil_loader加载图像
        return pil_loader(path)






class DatasetFolder(VisionDataset):
    """
    主要功能是从指定的根目录加载数据集，该目录下应该包含多个子文件夹，每个子文件夹代表一个类别。
    它会遍历这些子文件夹，将每个类别的图像文件及其对应的类别索引组合成一个样本列表。用户可以通过指定数据转换和目标转换函数，对加载的图像和对应的标签进行预处理。
    """
    def __init__(
            self,
            # 数据集的根目录
            root: str,
            # 用于加载图像文件的函数，根据文件路径返回图像对象
            loader: Callable[[str], Any],
            # 允许的文件扩展名元组，若为 None 则使用 is_valid_file 函数判断
            extensions: Optional[Tuple[str, ...]] = None,
            # 可选的转换函数，用于对图像进行预处理，如裁剪、缩放等
            transform: Optional[Callable] = None,
            # 可选的转换函数，用于对目标标签进行预处理
            target_transform: Optional[Callable] = None,
            # 可选的函数，用于检查文件是否为有效的图像文件
            is_valid_file: Optional[Callable[[str], bool]] = None,
            # 可选的索引文件路径，用于指定数据集的索引信息
            index_file: Optional[str] = None, 
    ) -> None:
        # 调用父类 VisionDataset 的构造函数，传入根目录、图像转换函数和目标标签转换函数
        super(DatasetFolder, self).__init__(
            root = root, 
            transform=transform,
            target_transform=target_transform)
        
        # 如果没有指定索引文件
        if index_file is None:
            # 调用 _find_classes 方法查找根目录下的所有类别，并返回类别列表和类别到索引的映射字典
            classes, class_to_idx = self._find_classes(self.root)
            # 生成样本列表，每一个元素是一个元组 (文件路径, 类别索引)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            # 如果生成的样本列表为空
            if len(samples) == 0:
                # 构造错误信息，提示在根目录的子文件夹中未找到有效文件
                msg = "Found 0 files in subfolders of: {}\n".format(self.root)
                # 如果指定了允许的文件扩展名，将其添加到错误信息中
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                # 抛出运行时错误，终止程序
                raise RuntimeError(msg)
        # 如果指定了索引文件
        else:
            # 以只读模式打开索引文件
            with open(index_file, mode="r", encoding="utf-8") as reader:
                # 初始化类别列表
                classes = []
                # 初始化索引数据字典
                index_data = {}
                # 逐行读取索引文件
                for line in reader:
                    # 将每行数据解析为 JSON 格式
                    data = json.loads(line)
                    # 获取类别名称
                    class_name = data["class"]
                    # 将类别名称添加到类别列表中
                    classes.append(class_name)
                    # 将类别名称和对应的文件列表存储到索引数据字典中
                    index_data[class_name] = data["files"]
                
                # 对类别列表进行排序
                classes.sort()
                # 生成类别名称到索引的映射字典
                class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
                # 初始化样本列表
                samples = []
                # 遍历索引数据字典中的每个类别
                for class_name in index_data:
                    # 获取当前类别的索引
                    class_index = class_to_idx[class_name]
                    # 遍历当前类别下的每个文件
                    for each_file in index_data[class_name]:
                        # 将文件的完整路径和对应类别索引组合成元组，并添加到样本列表中
                        samples.append(
                            (os.path.join(root, class_name, each_file), 
                            class_index)
                        )

        # 将加载器函数赋值给类的属性
        self.loader = loader
        # 将允许的文件扩展名元组赋值给类的属性
        self.extensions = extensions

        # 类别列表
        self.classes = classes
        # 类别名称：索引映射字典
        self.class_to_idx = class_to_idx
        # 将样本列表赋值给类的属性
        self.samples = samples
        # 从样本列表中提取所有样本的类别索引，存储到类的属性中
        self.targets = [s[1] for s in samples]

        # 打印找到的类别数量和样本数量
        print("Find %d classes and %d samples in root!" % (len(classes), len(samples)))

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        从指定目录中查找所有的类别，并生成类别名称到索引的映射。

        参数:
            dir (str): 要查找类别的目录路径。

        返回:
            Tuple[List[str], Dict[str, int]]: 一个元组，包含两个元素：
                - 第一个元素是一个列表，包含按字母顺序排序的类别名称。
                - 第二个元素是一个字典，将每个类别名称映射到一个唯一的整数索引。
        """
        # 遍历指定目录下的所有条目，筛选出所有的目录，并提取它们的名称作为类别名称
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # 对类别名称列表进行排序，确保每次运行时类别顺序一致
        classes.sort()
        # 生成一个字典，将每个类别名称映射到一个唯一的整数索引
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # 返回 class类别列表，class_to_idx类别名：索引的字典
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        从数据集中获取指定索引的样本和对应的目标标签。

        Args:
            index (int): 样本的索引

        Returns:
            tuple: (sample, target)，其中 target 是目标类别的类别索引。
        """
        # 循环尝试加载样本，直到成功为止
        while True:
            try:
                # 从样本列表中获取指定索引的文件路径和对应的类别索引
                path, target = self.samples[index]
                # 使用加载器函数加载图像文件
                sample = self.loader(path)
                # 若加载成功，跳出循环
                break
            except Exception as e:
                # 若加载过程中出现异常，打印异常信息
                print(e)
                # 随机选择一个新的索引，重新尝试加载样本
                index = random.randint(0, len(self.samples) - 1)

        # 如果定义了图像转换函数，则对加载的图像进行转换
        if self.transform is not None:
            sample = self.transform(sample)
        # 如果定义了目标标签转换函数，则对目标标签进行转换
        if self.target_transform is not None:
            target = self.target_transform(target)

        # 返回处理后的图像和目标标签
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]


def pil_loader(path: str) -> Image.Image:
    """
    使用PIL库加载图像文件，并将其转换为RGB格式。

    参数:
        path (str): 图像文件的路径。

    返回:
        Image.Image: 转换为RGB格式的PIL图像对象。
    """
    # 以二进制只读模式打开图像文件，避免ResourceWarning问题
    # 参考：https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        # 使用PIL的Image.open方法打开图像文件
        img = Image.open(f)
        # 将图像转换为RGB格式并返回
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    """
    使用accimage库加载图像文件。如果加载失败，将回退到使用PIL库加载。

    参数:
        path (str): 图像文件的路径。

    返回:
        Any: 加载的图像对象，可能是accimage.Image或PIL.Image对象。
    """
    import accimage
    try:
        # 尝试使用accimage库加载图像文件
        return accimage.Image(path)
    except IOError:
        # 若出现IO错误，可能是解码问题，回退到使用PIL库加载图像
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)







class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            # 数据集的根目录
            root: str,
            # 可选的转换函数，用于对图像进行预处理
            transform: Optional[Callable] = None,
            # 可选的转换函数，用于对目标标签进行预处理
            target_transform: Optional[Callable] = None,
            # 加载图像的函数，默认为 default_loader
            loader: Callable[[str], Any] = default_loader,
            # 可选的函数，用于检查文件是否为有效的图像文件
            is_valid_file: Optional[Callable[[str], bool]] = None,
            # 可选的索引文件路径
            index_file: Optional[str] = None, 
    ):
        # 如果没有提供 is_valid_file 函数，则使用 lambda 函数检查文件扩展名是否为允许的图像扩展名
        if is_valid_file is None:
            is_valid_file = lambda x: has_file_allowed_extension(x, IMG_EXTENSIONS)
        # 调用父类 DatasetFolder 的构造函数，传递必要的参数
        super(ImageFolder, self).__init__(root=root, loader=loader, 
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, 
                                          index_file=index_file)
        # 将父类的 samples 属性赋值给 imgs 属性，方便用户访问图像样本
        self.imgs = self.samples

































