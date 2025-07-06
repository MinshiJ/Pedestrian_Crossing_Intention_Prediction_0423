import pickle
import cv2

import numpy as np
import xml.etree.ElementTree as ET

from os.path import join, abspath, exists
from os import listdir, makedirs
from sklearn.model_selection import train_test_split, KFold


class WATCH_PED:
    def __init__(self, data_path='Watch_Ped/Watch_Ped_cache.pkl'):
        # Paths
        self._watch_ped_path = data_path
        # self.cache_path = join(self._watch_ped_path, 'cache')
        self._data_split_ids_path = join(self._watch_ped_path, 'video_splits')
        assert exists(self._watch_ped_path), \
            'Watch_Ped path does not exist: {}'.format(self._watch_ped_path)
        
    def generate_data_trajectory_sequence(self, image_set, **opts):
        params = {'fstride': 1,
                  'sample_type': 'all',  # 'beh'
                  'subset': 'default',
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'data_split_type': 'default',  # kfold, random, default
                  'seq_type': 'intention',
                  'min_track_size': 15,
                  'random_params': {'ratios': None,
                                    'val_data': True,
                                    'regen_data': False},
                  'kfold_params': {'num_folds': 5, 'fold': 1}}
        
        # Check for invalid options and print them
        invalid_opts = [k for k in opts.keys() if k not in params]
        if invalid_opts:
            print("Error: Invalid option(s) found: {}".format(invalid_opts))
            print("Valid options are: {}".format(list(params.keys())))
            raise ValueError("Wrong option(s): {}. Choose one of the following: {}".format(
                invalid_opts, list(params.keys())))
        
        params.update(opts)

        print('---------------------------------------------------------')
        print("Generating action sequence data")
        self._print_dict(params)

        annot_database = self.load_database()

        sequence = self._get_crossing(image_set, annot_database, **params)

        return sequence


    def _print_dict(self, dic):
        """
         Prints a dictionary, one key-value pair per line
         :param dic: Dictionary
         """
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))

    def load_database(self):
        print('---------------------------------------------------------')
        print("Loading database for Watch_Ped dataset")

        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self._watch_ped_path, 'Watch_Ped_cache.pkl')
        if exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('Watch_Ped database loaded from {}'.format(cache_file))
            return database
        else:
            raise FileNotFoundError('Watch_Ped database cache file not found: {}'.format(cache_file))
        

    def _bbox_dict_to_list(self, bbox_dict):
        """
        将字典格式的 bbox 转换为列表格式 [x1, y1, x2, y2]
        """
        if isinstance(bbox_dict, dict):
            return [
                bbox_dict.get('bbox_x1', 0),
                bbox_dict.get('bbox_y1', 0),
                bbox_dict.get('bbox_x2', 0),
                bbox_dict.get('bbox_y2', 0)
            ]
        return bbox_dict  # 如果已经是列表格式，直接返回
    
    def _ped_speed_dict_to_list(self, ped_speed_dict):
        """
        将字典格式的行人速度信息转换为列表格式 [vx, vy]
        """
        if isinstance(ped_speed_dict, dict):
            return [
                ped_speed_dict.get('acc_x', 0),
                ped_speed_dict.get('acc_y', 0),
                ped_speed_dict.get('acc_z', 0),
            ]
        return ped_speed_dict

    def _get_crossing(self, image_set, annotations, **params):
        print('---------------------------------------------------------')
        print("Generating crossing data")

        # seq_stride = params['fstride']
        # sq_ratio = params['squarify_ratio']
        
        box_seq = []
        depth_seq = []
        ped_speed_seq = []
        vehicle_seq = []
        crossing = []
        image_dimension = []

        video_ids, _pids = self._get_data_ids(image_set, params)
        
        # 添加调试信息
        print(f"Total video IDs loaded: {len(video_ids)}")
        
        valid_video_count = 0
        empty_video_count = 0
        no_bbox_count = 0

        for vid in sorted(video_ids):
            
            valid_ids = []

            img_annots = annotations[vid]['image_info']

            # 首先筛选出有效的bbox帧
            for frame_id, frame_data in img_annots.items():
                if 'bbox_ped' in frame_data:
                    bbox = frame_data['bbox_ped']
                    
                    if isinstance(bbox, dict):
                        coords = [bbox.get('bbox_x1', 0), bbox.get('bbox_y1', 0), 
                                bbox.get('bbox_x2', 0), bbox.get('bbox_y2', 0)]
                        if any(coord != 0 for coord in coords):
                            valid_ids.append(frame_id)

            # 统计信息
            if not img_annots:
                print(f"Video {vid}: No image annotations")
                continue
            
            if not valid_ids:
                print(f"Video {vid}: No valid bbox found (total frames: {len(img_annots)})")
                no_bbox_count += 1
                continue

            # 根据crossing状态截取valid_ids
            original_valid_count = len(valid_ids)
            if annotations[vid]['crossing'] == 1:
                crossing_point = annotations[vid]['crossing_point']
                # 只保留到crossing_point的帧
                valid_ids = [fid for fid in valid_ids if int(fid) <= crossing_point]
            else:
                # 如果没有crossing，去掉最后3帧（类似于end_idx = -3的逻辑）
                if len(valid_ids) > 3:
                    valid_ids = valid_ids[:-3]

            # 按帧ID排序
            valid_ids = sorted(valid_ids, key=lambda x: int(x))
            
            # 检查最终是否有有效帧
            if not valid_ids:
                print(f"Video {vid}: All frames filtered out (original: {original_valid_count}, crossing: {annotations[vid].get('crossing', 'N/A')})")
                empty_video_count += 1
                continue

            # 转换 bbox 为列表格式并添加到序列中
            bbox_list = []
            ped_speed_list = []
            for frame_id in valid_ids:
                bbox_dict = img_annots[frame_id]['bbox_ped']
                bbox_list.append(self._bbox_dict_to_list(bbox_dict))
                
                ped_speed_dict = img_annots[frame_id]['acc_info']
                ped_speed_list.append(self._ped_speed_dict_to_list(ped_speed_dict))

            box_seq.append(bbox_list)
            ped_speed_seq.append(ped_speed_list)
            depth_seq.append([img_annots[frame_id]['depth_info'] for frame_id in valid_ids])
            vehicle_seq.append([img_annots[frame_id]['vehicle_speed'] for frame_id in valid_ids])
            # 修复：将width和height作为一个元组传递给append
            image_dimension.append((annotations[vid]['width'], annotations[vid]['height']))
            crossing.append(annotations[vid]['crossing'])
            valid_video_count += 1

        print(f"Statistics:")
        print(f"  Total videos processed: {len(video_ids)}")
        print(f"  Valid videos with sequences: {valid_video_count}")
        print(f"  Videos with no valid bbox: {no_bbox_count}")
        print(f"  Videos filtered out completely: {empty_video_count}")
        print(f"  Final box_seq count: {len(box_seq)}")

        return {'bbox': box_seq,
                'ped_speed': ped_speed_seq,
                'depth': depth_seq,
                'vehicle_speed': vehicle_seq,
                'crossing': crossing,
                'image_dimension': image_dimension}
    
        # Trajectory data generation
    def _get_data_ids(self, image_set, params):
        vid_ids = []
        _pids = None
        sets = [image_set] if image_set != 'all' else ['train', 'test', 'val']
        for s in sets:
            vid_id_file = join(self._data_split_ids_path, s + '.txt')
            with open(vid_id_file, 'rt') as fid:
                vid_ids.extend([x.strip() for x in fid.readlines()])
        return vid_ids, _pids