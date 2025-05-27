from torchvision import transforms
import torch.nn.functional as F
import torch

def create_image_transform(img_size, is_train=True):
    """创建图像变换管道，针对模型需求优化"""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # 训练时的数据增强
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9), # 增强多样性
            transforms.ToTensor(),
            normalize,
        ])
    # 评估时的标准预处理
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform

def mixed_precision_collate(batch, fp16=True):
    """针对混合精度训练优化的批处理函数"""
    # 实现批数据整理，并根据需要转换为半精度
    # ...
