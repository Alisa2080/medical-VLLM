import random
import torch
import io
import pyarrow as pa
import os
from typing import Callable, Optional
from PIL import Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        image_augmentation: Callable,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=False,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
    ):
        """
        初始化 BaseDataset 类的实例。

        参数:
        data_dir (str): 数据集文件所在的目录，文件应为 .arrow 格式。
        image_augmentation (Callable): 图像增强器，直接接收PIL图像并返回增强后的tensor。
        image_size (int): 图像的大小。
        names (list): 数据集的名称列表。
        text_column_name (str, 可选): pyarrow 表中包含字符串列表的列名。默认为 ""。
        remove_duplicate (bool, 可选): 是否移除重复的文本。默认为 False。
        max_text_len (int, 可选): 文本的最大长度。默认为 40。
        draw_false_image (int, 可选): 抽取的虚假图像数量。默认为 0。
        draw_false_text (int, 可选): 抽取的虚假文本数量。默认为 0。
        image_only (bool, 可选): 是否仅使用图像数据。默认为 False。
        """
        # 调用父类的构造函数
        super().__init__()

        # 保存图像增强器
        self.image_augmentation = image_augmentation
        # 存储图像大小
        self.image_size = image_size
        # 存储文本列的名称
        self.text_column_name = text_column_name
        # 存储数据集的名称列表
        self.names = names
        # 存储最大文本长度
        self.max_text_len = max_text_len
        # 存储需要抽取的虚假图像数量
        self.draw_false_image = draw_false_image
        # 存储需要抽取的虚假文本数量
        self.draw_false_text = draw_false_text
        # 标记是否仅使用图像数据
        self.image_only = image_only
        # 存储数据集所在的目录
        self.data_dir = data_dir

        # 如果数据集名称列表不为空
        if len(names) != 0:
            # 读取所有可用的 .arrow 文件并转换为 pyarrow 表
            valid_names = []
            tables = []
            for name in names:
                file_path = f"{data_dir}/{name}.arrow"
                if os.path.isfile(file_path):
                    try:
                        table = pa.ipc.RecordBatchFileReader(pa.memory_map(file_path, "r")).read_all()
                        tables.append(table)
                        valid_names.append(name)
                    except:
                        print(f"{name}.arrow 读取失败")
                else:
                    print(f"{name}.arrow 不存在")
            names = valid_names
            self.names = names
        
            # 初始化表名列表
            self.table_names = list()
            # 遍历所有名称
            for i, name in enumerate(names):
                # 将每个表的名称重复添加到 table_names 列表中
                self.table_names += [name] * len(tables[i])

            # 将所有表合并为一个表
            self.table = pa.concat_tables(tables, promote_options='default')
            # 如果指定了文本列名称
            if text_column_name != "":
                # 存储文本列的名称
                self.text_column_name = text_column_name
                # 将文本列转换为 pandas 列表
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                # 如果需要移除重复文本
                self.all_texts = (
                    # 对每个文本列表进行去重
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    # 否则保持原样
                    else self.all_texts
                )
            else:
                # 如果未指定文本列名称，初始化空的文本列表
                self.all_texts = list()
        else:
            # 如果数据集名称列表为空，初始化空的文本列表
            self.all_texts = list()

        # 初始化索引映射字典
        self.index_mapper = dict()

        # 如果指定了文本列名称且不是仅使用图像数据
        if text_column_name != "" and not self.image_only:
            # 初始化索引计数器
            j = 0
            # 遍历所有文本列表
            for i, texts in enumerate(self.all_texts):
                # 遍历每个文本列表中的文本
                for _j in range(len(texts)):
                    # 将索引映射添加到 index_mapper 字典中
                    self.index_mapper[j] = (i, _j)
                    # 增加索引计数器
                    j += 1
        else:
            # 遍历所有表中的元素
            for i in range(len(self.table)):
                # 将索引映射添加到 index_mapper 字典中
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        """
        获取增强后的图像
        
        Args:
            index: 数据索引
            image_key: 图像列名
            
        Returns:
            dict: 包含增强后图像tensor和相关信息的字典
        """
        image = self.get_raw_image(index, image_key=image_key)
        
        # 使用增强器处理图像
        try:
            image_tensor = self.image_augmentation(image)
            
            # 确保返回的是tensor，如果增强器返回多个视图，取第一个
            if isinstance(image_tensor, (list, tuple)):
                image_tensor = image_tensor[0]
            
            # 如果需要包装成列表以保持与原有接口的兼容性
            if not isinstance(image_tensor, list):
                image_tensor = [image_tensor]
                
        except Exception as e:
            print(f"Image augmentation failed for index {index}: {e}")
            # 降级处理：简单resize和normalize
            import torchvision.transforms as T
            fallback_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = [fallback_transform(image)]
        
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        """
        获取随机的虚假图像
        
        Args:
            rep: 重复次数标识
            image_key: 图像列名
            
        Returns:
            dict: 包含虚假图像tensor的字典
        """
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        
        # 使用增强器处理虚假图像
        try:
            image_tensor = self.image_augmentation(image)
            
            # 确保返回的是tensor，如果增强器返回多个视图，取第一个
            if isinstance(image_tensor, (list, tuple)):
                image_tensor = image_tensor[0]
            
            # 包装成列表以保持兼容性
            if not isinstance(image_tensor, list):
                image_tensor = [image_tensor]
                
        except Exception as e:
            print(f"False image augmentation failed: {e}")
            # 降级处理
            import torchvision.transforms as T
            fallback_transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = [fallback_transform(image)]
        
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def get_text_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

        for img_key in img_keys:
            new_imgs = [tmp_img[0] for tmp_img in dict_batch[img_key]]
            batch_new_imgs = torch.stack(new_imgs, dim=0)
            dict_batch[img_key] = [batch_new_imgs]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"], dtype=torch.bool),  # 修改：强制使用布尔类型
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask.bool()  # 修改：确保attention_mask是布尔类型

        return dict_batch