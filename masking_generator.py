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
            mask_schedule_steps=0 
            ):

        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.step = 0

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

    def __call__(self):
        target_masks = self.num_masking_patches

        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        cnt = 0
        if target_masks > 0.8*self.num_patches:
            flat = torch.rand(self.num_patches) < (target_masks/self.num_patches)
            mask = flat.view(self.height, self.width)
        else:
            max_attempts = 100
            attempts = 0
            while cnt < target_masks and attempts < max_attempts:
                attempts += 1
                area = random.uniform(self.min_num_patches, self.max_num_patches)
                ar = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(area*ar)))
                w = int(round(math.sqrt(area/ar)))
                if h>=1 and w>=1 and h<=self.height and w<=self.width:
                    top = random.randint(0, self.height-h)
                    left = random.randint(0, self.width-w)
                    block = mask[top:top+h, left:left+w]
                    free = (~block).nonzero(as_tuple=False)
                    if free.numel()>0:
                        k = min(free.size(0), target_masks-cnt)
                        idx = free[torch.randperm(free.size(0))[:k]]
                        mask[top+idx[:,0], left+idx[:,1]] = True
                        cnt += k

            diff = cnt - target_masks
            if diff>0:
                pos = mask.nonzero(as_tuple=False)
                sel = pos[torch.randperm(pos.size(0))[:diff]]
                mask[sel[:,0], sel[:,1]] = False
            elif diff<0:
                pos = (~mask).nonzero(as_tuple=False)
                sel = pos[torch.randperm(pos.size(0))[:(-diff)]]
                mask[sel[:,0], sel[:,1]] = True
        
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