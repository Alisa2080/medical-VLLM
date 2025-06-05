from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)
from .square_transform import (
    square_transform,
    square_transform_randaug,
)
from .Pathogram_Transformation import (
    PathologyAugmentation,
    MultiViewPathologyAugmentation,
)

# ==================== 废弃的转换系统 ====================
# 注意：以下函数和字典已废弃，不再用于BaseDataset
# 保留仅为了向后兼容性，建议使用 PathologyAugmentation 类
_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "square_transform": square_transform,
    "square_transform_randaug": square_transform_randaug,
}

def keys_to_transforms(keys: list, size=224):
    """
    DEPRECATED: 此函数已废弃，不再用于BaseDataset
    
    请使用 PathologyAugmentation 类替代：
    
    Example:
        # 旧方式（已废弃）
        transforms = keys_to_transforms(["square_transform"], size=384)
        
        # 新方式（推荐）
        from transforms.Pathogram_Transformation import PathologyAugmentation
        config = {"input_size": 384, "enable_pathology_augmentation": True, ...}
        augmentation = PathologyAugmentation(config, is_training=True)
    """
    import warnings
    warnings.warn(
        "keys_to_transforms is deprecated and will be removed in a future version. "
        "Use PathologyAugmentation class instead for better pathology image support.",
        DeprecationWarning,
        stacklevel=2
    )
    return [_transforms[key](size=size) for key in keys]

# ==================== 新的增强系统导出 ====================
__all__ = [
    "PathologyAugmentation",
    "MultiViewPathologyAugmentation",
    # 向后兼容
    "pixelbert_transform",
    "pixelbert_transform_randaug", 
    "square_transform",
    "square_transform_randaug",
    "keys_to_transforms",  # 废弃但保留
]