import h5py
import numpy as np
import os
import pdb
from wsi_core.util_classes import Mosaic_Canvas
from PIL import Image
import math
import cv2
from tqdm import tqdm
import openslide
from typing import Tuple, Optional, Dict, Any
import math

class WSIAdaptiveParameterEngine:
    """
    WSI自适应参数推荐引擎
    根据WSI特性智能推荐patch_level和step_size
    """
    
    def __init__(self, 
                 target_patch_mpp: float = 0.75,
                 target_patch_physical_size: Optional[int] = None,
                 min_patches_target: int = 300,
                 max_patches_target: int = 2500,
                 default_wsi_mpp: float = 0.25,
                 min_step_size_ratio: float = 0.25,
                 max_iterations: int = 5):
        """
        初始化参数推荐引擎
        
        Args:
            target_patch_mpp: 目标patch的MPP值（微米/像素）
            target_patch_physical_size: 目标patch覆盖的物理尺寸（微米）
            min_patches_target: 目标最小patch数量
            max_patches_target: 目标最大patch数量
            default_wsi_mpp: 默认WSI的MPP值
            min_step_size_ratio: step_size相对于patch_size的最小比例
            max_iterations: 参数调整的最大迭代次数
        """
        self.target_patch_mpp = target_patch_mpp
        self.target_patch_physical_size = target_patch_physical_size
        self.min_patches_target = min_patches_target
        self.max_patches_target = max_patches_target
        self.default_wsi_mpp = default_wsi_mpp
        self.min_step_size_ratio = min_step_size_ratio
        self.max_iterations = max_iterations
    
    def get_wsi_mpp(self, wsi_object) -> float:
        """获取WSI的MPP值"""
        try:
            wsi = wsi_object.getOpenSlide()
            mpp_x = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
            mpp_y = wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            
            if mpp_x is not None and mpp_y is not None:
                return (float(mpp_x) + float(mpp_y)) / 2.0
            else:
                print(f"Warning: MPP not found in WSI properties. Using default MPP: {self.default_wsi_mpp}")
                return self.default_wsi_mpp
        except Exception as e:
            print(f"Warning: Error reading MPP from WSI: {e}. Using default MPP: {self.default_wsi_mpp}")
            return self.default_wsi_mpp
    
    def estimate_tissue_area(self, wsi_object, seg_level: int = None) -> Tuple[float, int]:
        """
        估算组织区域面积
        
        Returns:
            Tuple[float, int]: (level0坐标系下的组织面积, 使用的seg_level)
        """
        wsi = wsi_object.getOpenSlide()
        
        # 选择合适的分割层级
        if seg_level is None:
            seg_level = wsi.get_best_level_for_downsample(64)
        
        # 快速组织分割
        try:
            img = np.array(wsi.read_region((0, 0), seg_level, wsi_object.level_dim[seg_level]))
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)
            _, img_otsu = cv2.threshold(img_med, 8, 255, cv2.THRESH_BINARY)
            
            # 查找轮廓
            contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) == 0:
                return 0.0, seg_level
            
            # 计算组织面积（在seg_level坐标系下）
            tissue_area_seg_level = 0
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:] if hierarchy is not None else None
            
            if hierarchy is not None:
                # 找到前景轮廓（parent == -1）
                foreground_indices = np.flatnonzero(hierarchy[:, 1] == -1)
                for cont_idx in foreground_indices:
                    cont = contours[cont_idx]
                    area = cv2.contourArea(cont)
                    if area > 100:  # 过滤小的伪影
                        # 减去孔洞面积
                        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                        hole_areas = sum(cv2.contourArea(contours[hole_idx]) for hole_idx in holes)
                        tissue_area_seg_level += (area - hole_areas)
            else:
                # 简单情况：没有层次结构
                tissue_area_seg_level = sum(cv2.contourArea(cont) for cont in contours if cv2.contourArea(cont) > 100)
            
            # 转换到level 0坐标系
            scale = wsi_object.level_downsamples[seg_level]
            tissue_area_level0 = tissue_area_seg_level * scale[0] * scale[1]
            
            return tissue_area_level0, seg_level
            
        except Exception as e:
            print(f"Warning: Error estimating tissue area: {e}")
            # 返回一个保守估计
            w, h = wsi_object.level_dim[0]
            return w * h * 0.3, seg_level  # 假设30%是组织
    
    def recommend_patch_level(self, wsi_object, patch_size: int, wsi_mpp: float) -> int:
        """
        推荐最佳patch_level
        
        Args:
            wsi_object: WSI对象
            patch_size: patch尺寸（像素）
            wsi_mpp: WSI的MPP值
        
        Returns:
            推荐的patch_level
        """
        available_levels = len(wsi_object.level_dim)
        wsi = wsi_object.getOpenSlide()
        
        best_level = 0
        best_score = float('inf')
        
        for level in range(available_levels):
            downsample = wsi_object.level_downsamples[level]
            level_mpp = wsi_mpp * downsample[0]  # 该层级的MPP
            
            # 策略1: 基于目标MPP
            if self.target_patch_mpp is not None:
                mpp_score = abs(level_mpp - self.target_patch_mpp)
            else:
                mpp_score = 0
            
            # 策略2: 基于目标物理尺寸
            if self.target_patch_physical_size is not None:
                physical_size = patch_size * level_mpp
                physical_size_score = abs(physical_size - self.target_patch_physical_size) / self.target_patch_physical_size
            else:
                physical_size_score = 0
            
            # 策略3: 避免过高或过低分辨率
            w, h = wsi_object.level_dim[level]
            if w * h < 1000 * 1000:  # 太小的层级
                resolution_penalty = 2.0
            elif w * h > 50000 * 50000:  # 太大的层级
                resolution_penalty = 1.0
            else:
                resolution_penalty = 0
            
            # 综合评分
            total_score = mpp_score + physical_size_score + resolution_penalty
            
            if total_score < best_score:
                best_score = total_score
                best_level = level
        
        print(f"Recommended patch_level: {best_level} (MPP: {wsi_mpp * wsi_object.level_downsamples[best_level][0]:.3f})")
        return best_level
    
    def recommend_step_size(self, wsi_object, patch_level: int, patch_size: int, tissue_area_level0: float) -> Tuple[int, int]:
        """
        推荐最佳step_size
        
        Args:
            wsi_object: WSI对象
            patch_level: patch层级
            patch_size: patch尺寸
            tissue_area_level0: level0坐标系下的组织面积
        
        Returns:
            Tuple[int, int]: (推荐的step_size, 调整后的patch_level)
        """
        current_patch_level = patch_level
        
        for iteration in range(self.max_iterations):
            # 计算当前层级下的组织面积
            downsample = wsi_object.level_downsamples[current_patch_level]
            tissue_area_current_level = tissue_area_level0 / (downsample[0] * downsample[1])
            
            # 估算patch数量（假设step_size = patch_size）
            estimated_patches = tissue_area_current_level / (patch_size * patch_size)
            
            print(f"Iteration {iteration + 1}: patch_level={current_patch_level}, estimated_patches={estimated_patches:.0f}")
            
            # 检查patch数量是否在目标范围内
            if self.min_patches_target <= estimated_patches <= self.max_patches_target:
                # 数量合适，使用默认step_size
                step_size = patch_size
                print(f"Patch count in target range. Using step_size = patch_size = {step_size}")
                return step_size, current_patch_level
            
            elif estimated_patches < self.min_patches_target:
                # patch太少，需要增加重叠或降低patch_level
                if iteration < self.max_iterations - 1 and current_patch_level > 0:
                    # 尝试降低patch_level（提高分辨率）
                    current_patch_level -= 1
                    print(f"Too few patches. Reducing patch_level to {current_patch_level}")
                    continue
                else:
                    # 增加重叠
                    target_ratio = self.min_patches_target / estimated_patches
                    step_size_ratio = 1.0 / math.sqrt(target_ratio)
                    step_size_ratio = max(step_size_ratio, self.min_step_size_ratio)
                    step_size = max(int(patch_size * step_size_ratio), int(patch_size * self.min_step_size_ratio))
                    print(f"Increasing overlap. Using step_size = {step_size}")
                    return step_size, current_patch_level
            
            else:  # estimated_patches > max_patches_target
                # patch太多，需要减少重叠或提高patch_level
                if iteration < self.max_iterations - 1 and current_patch_level < len(wsi_object.level_dim) - 1:
                    # 尝试提高patch_level（降低分辨率）
                    current_patch_level += 1
                    print(f"Too many patches. Increasing patch_level to {current_patch_level}")
                    continue
                else:
                    # 减少重叠（step_size = patch_size已经是最大值）
                    step_size = patch_size
                    print(f"Using maximum step_size = {step_size} (no overlap)")
                    return step_size, current_patch_level
        
        # 如果迭代结束仍未找到最优解，返回保守设置
        step_size = patch_size
        print(f"Max iterations reached. Using step_size = {step_size}")
        return step_size, current_patch_level
    
    def recommend_parameters(self, wsi_object, patch_size: int, user_patch_level: Optional[int] = None, user_step_size: Optional[int] = None) -> Dict[str, Any]:
        """
        为WSI推荐最佳参数
        
        Args:
            wsi_object: WSI对象
            patch_size: patch尺寸
            user_patch_level: 用户指定的patch_level（如果提供则不调整）
            user_step_size: 用户指定的step_size（如果提供则不调整）
        
        Returns:
            推荐参数字典
        """
        print(f"\n=== Analyzing WSI: {wsi_object.name} ===")
        
        # 获取WSI基础信息
        wsi_mpp = self.get_wsi_mpp(wsi_object)
        tissue_area_level0, seg_level_used = self.estimate_tissue_area(wsi_object)
        
        print(f"WSI MPP: {wsi_mpp:.3f} µm/pixel")
        print(f"Estimated tissue area: {tissue_area_level0:.0f} pixels² (level 0)")
        print(f"Segmentation performed at level: {seg_level_used}")
        
        # 推荐patch_level
        if user_patch_level is not None:
            recommended_patch_level = user_patch_level
            print(f"Using user-specified patch_level: {recommended_patch_level}")
        else:
            recommended_patch_level = self.recommend_patch_level(wsi_object, patch_size, wsi_mpp)
        
        # 推荐step_size
        if user_step_size is not None:
            recommended_step_size = user_step_size
            print(f"Using user-specified step_size: {recommended_step_size}")
        else:
            recommended_step_size, recommended_patch_level = self.recommend_step_size(
                wsi_object, recommended_patch_level, patch_size, tissue_area_level0
            )
        
        # 计算最终预期的patch数量
        downsample = wsi_object.level_downsamples[recommended_patch_level]
        tissue_area_patch_level = tissue_area_level0 / (downsample[0] * downsample[1])
        expected_patches = tissue_area_patch_level / (recommended_step_size * recommended_step_size)
        
        # 计算实际的MPP和物理覆盖范围
        actual_mpp = wsi_mpp * downsample[0]
        actual_physical_size = patch_size * actual_mpp
        
        result = {
            'patch_level': recommended_patch_level,
            'step_size': recommended_step_size,
            'expected_patches': int(expected_patches),
            'actual_mpp': actual_mpp,
            'actual_physical_size': actual_physical_size,
            'tissue_area_level0': tissue_area_level0,
            'wsi_mpp': wsi_mpp,
            'recommendations_applied': {
                'patch_level_adjusted': user_patch_level is None,
                'step_size_adjusted': user_step_size is None
            }
        }
        
        print(f"\n=== Recommended Parameters ===")
        print(f"patch_level: {result['patch_level']}")
        print(f"step_size: {result['step_size']}")
        print(f"Expected patches: {result['expected_patches']}")
        print(f"Actual patch MPP: {result['actual_mpp']:.3f} µm/pixel")
        print(f"Actual physical coverage: {result['actual_physical_size']:.1f} µm x {result['actual_physical_size']:.1f} µm")
        print("=" * 40)
        
        return result
def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt':(x,y)})
                yield process_dict
            else:
                yield (x,y)

def savePatchIter_bag_hdf5(patch):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs', 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=True):
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
        
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    for idx in tqdm(range(total)):        
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['imgs']
        coords = file['coords'][:]
        if 'downsampled_level_dim' in dset.attrs.keys():
            w, h = dset.attrs['downsampled_level_dim']
        else:
            w, h = dset.attrs['level_dim']

    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print(f'number of patches: {len(coords)}')
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    return heatmap

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    wsi = wsi_object.getOpenSlide()
    w, h = wsi.level_dimensions[0]
    print('original size: {} x {}'.format(w, h))
    
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('downscaled size for stiching: {} x {}'.format(w, h))

    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        print('start stitching {}'.format(dset.attrs['name']))
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']
    
    print(f'number of patches: {len(coords)}')
    print(f'patch size: {patch_size} x {patch_size} patch level: {patch_level}')
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print(f'ref patch size: {patch_size} x {patch_size}')

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    return heatmap

def SamplePatches(coords_file_path, save_file_path, wsi_object, 
    patch_level=0, custom_downsample=1, patch_size=256, sample_num=100, seed=1, stitch=True, verbose=1, mode='w'):
    
    with h5py.File(coords_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        h5_patch_size = dset.attrs['patch_size']
        h5_patch_level = dset.attrs['patch_level']
    
    if verbose>0:
        print('in .h5 file: total number of patches: {}'.format(len(coords)))
        print('in .h5 file: patch size: {}x{} patch level: {}'.format(h5_patch_size, h5_patch_size, h5_patch_level))

    if patch_level < 0:
        patch_level = h5_patch_level

    if patch_size < 0:
        patch_size = h5_patch_size

    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(coords)), min(len(coords), sample_num), replace=False)

    target_patch_size = np.array([patch_size, patch_size])
    
    if custom_downsample > 1:
        target_patch_size = (np.array([patch_size, patch_size]) / custom_downsample).astype(np.int32)
        
    if stitch:
        canvas = Mosaic_Canvas(patch_size=target_patch_size[0], n=sample_num, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1)
    else:
        canvas = None
    
    for idx in indices:
        coord = coords[idx]
        patch = wsi_object.wsi.read_region(coord, patch_level, tuple([patch_size, patch_size])).convert('RGB')
        if custom_downsample > 1:
            patch = patch.resize(tuple(target_patch_size))

        # if isBlackPatch_S(patch, rgbThresh=20, percentage=0.05) or isWhitePatch_S(patch, rgbThresh=220, percentage=0.25):
        #     continue

        if stitch:
            canvas.paste_patch(patch)

        asset_dict = {'imgs': np.array(patch)[np.newaxis,...], 'coords': coord}
        save_hdf5(save_file_path, asset_dict, mode=mode)
        mode='a'

    return canvas, len(coords), len(indices)