import random
import math
import numpy as np
import torch

class MaskingGenerator:
    def __init__(
            self,
            input_size,
            num_masking_patches,
            min_num_patches=16,
            max_num_patches=75,
            min_aspect=0.3,
            max_aspect=None,
            block_wise=True,
            ):

        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        self.block_wise = block_wise

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta
    
    def _generate_block_mask(self, target_masks):
        """生成块状掩码，适合病理图像"""
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 100
        
        while masked_count < target_masks and max_attempts > 0:
            # 计算剩余需要掩码的数量
            remaining = target_masks - masked_count
            
            # 动态调整块大小
            target_area = min(remaining, 
                            random.uniform(self.min_num_patches, 
                                         min(self.max_num_patches, remaining * 1.5)))
            
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            # 确保块不会超出边界
            if h <= self.height and w <= self.width:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                # 计算这个区域中未掩码的patch数量
                region = mask[top:top+h, left:left+w]
                free_patches = (~region).sum().item()
                
                if free_patches > 0:
                    # 只掩码未被掩码的patch，避免重复
                    mask[top:top+h, left:left+w] = True
                    masked_count = mask.sum().item()
            
            max_attempts -= 1
        
        # 精确调整到目标数量
        current_count = mask.sum().item()
        if current_count > target_masks:
            # 随机移除多余的掩码
            masked_positions = mask.nonzero(as_tuple=False)
            remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
            for idx in remove_indices:
                pos = masked_positions[idx]
                mask[pos[0], pos[1]] = False
        elif current_count < target_masks:
            # 随机添加不足的掩码
            unmasked_positions = (~mask).nonzero(as_tuple=False)
            if len(unmasked_positions) > 0:
                add_count = min(target_masks - current_count, len(unmasked_positions))
                add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                for idx in add_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
        
        return mask
    
    def _generate_random_mask(self, target_masks):
        """生成随机掩码，适合高掩码率情况"""
        prob = target_masks / self.num_patches
        mask = torch.rand(self.height, self.width) < prob
        
        # 精确调整到目标数量
        current_count = mask.sum().item()
        if current_count != target_masks:
            if current_count > target_masks:
                masked_positions = mask.nonzero(as_tuple=False)
                remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
                for idx in remove_indices:
                    pos = masked_positions[idx]
                    mask[pos[0], pos[1]] = False
            else:
                unmasked_positions = (~mask).nonzero(as_tuple=False)
                if len(unmasked_positions) > 0:
                    add_count = min(target_masks - current_count, len(unmasked_positions))
                    add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                    for idx in add_indices:
                        pos = unmasked_positions[idx]
                        mask[pos[0], pos[1]] = True
        
        return mask
    
    def __call__(self):
        target_masks = self.num_masking_patches
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        
        if self.block_wise:
            # 改进的块状掩码生成
            mask = self._generate_block_mask(target_masks)
        else:
            # 随机掩码生成（适用于高掩码率）
            mask = self._generate_random_mask(target_masks)
        
        return mask
        
if __name__ == '__main__':
    import pdb
    generator = MaskingGenerator(input_size=14, num_masking_patches=118, min_num_patches=16,)
    for i in range(10):
        mask = generator()
        if mask.sum() != 118:
            pdb.set_trace()
            print(mask)
            print(mask.sum())