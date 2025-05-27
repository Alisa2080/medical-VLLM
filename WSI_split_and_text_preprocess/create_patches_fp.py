# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
from wsi_core.wsi_utils import WSIAdaptiveParameterEngine


def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed



def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 384, step_size = 384, 
				  seg_params = {'seg_level': 1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = True, auto_skip=True, process_list = None,
				  auto_adjust_params = False,
                  target_patch_mpp = None,
                  target_patch_physical_size = None,
                  min_patches_target = 300,
                  max_patches_target = 2500,
                  default_wsi_mpp = 0.25):
	


	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	if auto_adjust_params:
		param_engine = WSIAdaptiveParameterEngine(
            target_patch_mpp=target_patch_mpp,
            target_patch_physical_size=target_patch_physical_size,
            min_patches_target=min_patches_target,
            max_patches_target=max_patches_target,
            default_wsi_mpp=default_wsi_mpp
        )
		print("Adaptive parameter adjustment enabled")
	else:
		param_engine = None

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		slide_patch_folder_path = os.path.join(patch_save_dir, slide_id)
		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue
		if os.path.isdir(slide_patch_folder_path) and len(os.listdir(slide_patch_folder_path)) > 0 :
			print('Patches for {} already exist in destination location {}, skipped'.format(slide_id, slide_patch_folder_path))
			df.loc[idx, 'status'] = 'already_exist'
			continue
		# Inialize WSI
		full_path = os.path.join(source, slide)
		try:
			WSI_object = WholeSlideImage(full_path)
		except Exception as e:
			print(f"Error initializing WSI object for {slide}: {e}")
			df.loc[idx, 'status'] = 'failed_wsi_load'
			continue

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level
		

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']
		print(f"INFO: For slide {slide_id}, final seg_level to be used: {current_seg_params['seg_level']}")
		
		w, h = WSI_object.level_dim[current_seg_params['seg_level']]
		
		# 自适应参数调整
		if auto_adjust_params and param_engine:
			try:
                # 检查用户是否已指定参数
				user_patch_level = None
				user_step_size = None
                
                # 如果用户明确指定了patch_level或overlap，则不调整
				if patch_level != 0:  # 用户指定了非默认patch_level
					user_patch_level = patch_level
                
                # 检查是否用户通过overlap指定了step_size
                # 这里假设如果step_size != patch_size，说明用户有意设置
				if step_size != patch_size:
					user_step_size = step_size
                
                # 获取推荐参数
				recommendations = param_engine.recommend_parameters(
                    WSI_object, patch_size, user_patch_level, user_step_size
                )
                
                # 应用推荐参数
				adaptive_patch_level = recommendations['patch_level']
				adaptive_step_size = recommendations['step_size']
                
                # 记录推荐信息到DataFrame
				df.loc[idx, 'recommended_patch_level'] = adaptive_patch_level
				df.loc[idx, 'recommended_step_size'] = adaptive_step_size
				df.loc[idx, 'expected_patches'] = recommendations['expected_patches']
				df.loc[idx, 'actual_mpp'] = recommendations['actual_mpp']
				df.loc[idx, 'actual_physical_size'] = recommendations['actual_physical_size']
                
				print(f"Applied adaptive parameters: patch_level={adaptive_patch_level}, step_size={adaptive_step_size}")
                
			except Exception as e:
				print(f"Warning: Error in adaptive parameter adjustment for {slide_id}: {e}")
				print("Using original parameters")
				adaptive_patch_level = patch_level
				adaptive_step_size = step_size
		else:
			adaptive_patch_level = patch_level
			adaptive_step_size = step_size
		
		seg_time_elapsed = -1
		if seg:
			try:
				WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 
			except Exception as e_seg:
				print(f"Error during segmentation for {slide_id}: {e_seg}")
				df.loc[idx, 'status'] = 'failed_seg'
				continue

		if save_mask:
			try:
				mask_vis = WSI_object.visWSI(**current_vis_params) # Corrected variable name
				mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
				mask_vis.save(mask_path) # Corrected variable name
			except Exception as e_mask:
				print(f"Error saving mask for {slide_id}: {e_mask}")

		patch_time_elapsed = -1 # Default time
		if patch:
			patch_start_time = time.time()
			    # Ensure all necessary parameters for process_contours_and_save_patches are passed
			try:    # from current_patch_params or defaults.
				num_patches_saved = WSI_object.process_contours_and_save_patches(
                    patch_output_root_dir=patch_save_dir, # This is the root dir for all slide patch folders
                    patch_level=adaptive_patch_level,  # 使用自适应参数
                    patch_size=patch_size,
                    step_size=adaptive_step_size,  # 使用自适应参数
                    contour_fn=current_patch_params.get('contour_fn', 'four_pt'),
                    use_padding=current_patch_params.get('use_padding', True),
                    white_thresh=current_patch_params.get('white_thresh', 15),
                    black_thresh=current_patch_params.get('black_thresh', 50),
                    white_black_filter=current_patch_params.get('white_black_filter', True)
                )
				patch_time_elapsed = time.time() - patch_start_time
				
				# 记录实际生成的patch数量
				df.loc[idx, 'actual_patches'] = num_patches_saved

				if num_patches_saved == 0 and WSI_object.contours_tissue and len(WSI_object.contours_tissue) > 0:
					print(f"Warning: No patches were saved for {slide_id} despite having tissue contours.")
                    # Optionally set a specific status if no patches are saved from valid contours
                    # df.loc[idx, 'status'] = 'no_patches_saved' 
			except Exception as e_patch:
				print(f"Error during patching for {slide_id}: {e_patch}")
				df.loc[idx, 'status'] = 'failed_patching'
				patch_time_elapsed = time.time() - patch_start_time # record time even if error
                # continue # Decide if you want to continue to next slide or try stitching
		

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
# parser.add_argument('--step_size', type = int, default=256,
					# help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--overlap',type=float,default=0.0,help="patch 重叠比例(0.0 表示无重叠)")
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

# 新增自适应参数控制选项
parser.add_argument('--auto_adjust_params', default=False, action='store_true',
                    help='enable adaptive parameter adjustment')
parser.add_argument('--target_patch_mpp', type=float, default=0.75,
                    help='target patch MPP (micrometers per pixel)')
parser.add_argument('--target_patch_physical_size', type=int, default=None,
                    help='target patch physical size in micrometers')
parser.add_argument('--min_patches_target', type=int, default=300,
                    help='minimum target number of patches per WSI')
parser.add_argument('--max_patches_target', type=int, default=2500,
                    help='maximum target number of patches per WSI')
parser.add_argument('--default_wsi_mpp', type=float, default=0.25,
                    help='default WSI MPP when not available in metadata')


if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level':1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)
	if args.overlap == 0.0:
		step_size = args.patch_size
	else:
		step_size = int(args.patch_size * (1 - args.overlap))
	print(f"patch_size:{args.patch_size},step_size:{step_size}")
	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip,
											auto_adjust_params=args.auto_adjust_params,
											target_patch_mpp=args.target_patch_mpp,
											target_patch_physical_size=args.target_patch_physical_size,
											min_patches_target=args.min_patches_target,
											max_patches_target=args.max_patches_target,
											default_wsi_mpp=args.default_wsi_mpp)
