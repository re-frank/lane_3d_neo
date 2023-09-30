import numpy as np
from scipy.spatial.transform import Rotation as R
import os.path as osp
import os
import pickle
import glob
from base_dataset import BaseDatasets



class Datasets(BaseDatasets):
    '''
        定义Neolix数据集的内容
    '''
    def __init__(self, data_dir, info_dir, special_scene={}) -> None:
        self.data_dir = data_dir        # 数据所在文件夹
        self.info_dir = info_dir
        self.dataset = 'Neolix'
        self.description_dict = special_scene
    
    def _get_description(self, meta_dict):
        '''
            get scene description
            (including vehicle_id, date, weather, site, lighting, scene)
        '''

        description_list = [ 
            meta_dict['date'], 
            meta_dict['site'],
            meta_dict['vehicle_id'],
            meta_dict['lighting'],
            meta_dict['weather'], 
            meta_dict['scene']
            ]
        
        return description_list

    
    def get_data(self):
        with open(self.info_dir, 'rb') as f:
            data_info = pickle.load(f)
        scenes_data = {
            'dataset': self.dataset,
            'scenes': [],
            'description_dict': self.description_dict
        }
        # 遍历每个场景(meta)
        for scene_info in data_info:
            scene_data = {}
            scene_data['scene_name'] = scene_info['meta']['record_id']
            # 尝试读取离线数据
            if scene_data['scene_name'] != '20230720121852.record.00007':
                continue
            scene_data_npy_name = 'data/Neolix/offline_Neolix/' + scene_data['scene_name'] + '.npy'

            if os.path.exists(scene_data_npy_name):
                scene_data = np.load(scene_data_npy_name, allow_pickle=True).item()
            else:
                scene_data['samples'] = []
                scene_data['description'] = self._get_description(scene_info['meta'])
                # 判断是否生成位姿
                sensor_list = scene_info['sweeps'].keys()
                for i, top_lidar_info in enumerate(scene_info['sweeps']['top_lidar']):
                    sample_data = {
                        'is_key_frame': False,
                        'data': [],
                        'annos': []
                    }
                    # 添加雷达信息
                    main_lidar_name = osp.splitext(top_lidar_info.split('/')[-1])[0]
                    # 读取对应pose文件
                    pose_file_path = osp.join(
                        self.data_dir, 
                        'to_anno',
                        scene_data['scene_name'], 
                        'transformed_dr', 
                        main_lidar_name + '_*.bin'
                        )
                    pose_file = glob.glob(pose_file_path)
                    if len(pose_file) != 0:
                        # pose矩阵生成
                        pose = np.fromfile(pose_file[0], dtype=np.float64, count=-1)
                        pose_trans = np.array(pose[:3]).reshape((3,1))
                        pose_rota = R.from_quat(pose[3:]).as_matrix()
                        pose_to_map = np.identity(4)
                        pose_to_map[:3, :3] = pose_rota
                        pose_to_map[:3, 3] = pose_trans.T
                    # 顶部雷达信息
                    top_lidar_data = {
                        'sensor_name': 'top_lidar',
                        'sensor_type': 'lidar',
                        'timestamp': int(''.join(main_lidar_name.split('.'))[:-3]),
                        'filename': osp.join('to_anno', top_lidar_info),
                        'calibration': {
                            'extrinsic_rotation': np.identity(3),
                            'extrinsic_translation': np.zeros((3,1))
                        },
                        'pose': {
                            'rotation_to_map': pose_rota,
                            'translation_to_map': pose_trans,
                            'rotation_to_sensor': np.linalg.inv(pose_to_map)[:3, :3],
                            'translation_to_sensor': np.linalg.inv(pose_to_map)[:3, 3].reshape((3,1))
                        } if len(pose_file) != 0 else None
                    }
                    sample_data['data'].append(top_lidar_data)
                    # 其他传感器
                    for sensor_name in sensor_list:
                        if sensor_name == 'top_lidar':
                            continue
                        sensor_type = 'lidar' if 'lidar' in sensor_name else 'cam'
                        # 获取激光雷达数据对应的其他传感器数据
                        if main_lidar_name in scene_info['sweeps'][sensor_name][i]:
                            sensor_file = scene_info['sweeps'][sensor_name][i]
                        else:
                            sensor_file = [s for s in scene_info['sweeps'][sensor_name] if main_lidar_name in s][0]
                        sensor_time = ''.join(osp.splitext(sensor_file.split('_')[-1])[0].split('.'))[:-3]
                        # 数据按雷达/图像分别处理
                        if sensor_type == 'lidar':
                            extrinsic_rotation = np.array(scene_info['calib']['lidar_rotation'][sensor_name]).reshape((3,3))
                            extrinsic_translation = np.array(scene_info['calib']['lidar_translation'][sensor_name]).reshape((3,1))
                            if len(pose_file) != 0:
                                r_sensor_to_map = pose_rota @ extrinsic_rotation
                                t_sensor_to_map = pose_rota @ extrinsic_translation + pose_trans
                                sensor_to_map = np.identity(4)
                                sensor_to_map[:3, :3] = r_sensor_to_map
                                sensor_to_map[:3, 3] = t_sensor_to_map.T
                            sensor_data = {
                                'sensor_name': sensor_name,
                                'sensor_type': sensor_type,
                                'timestamp': int(sensor_time),
                                'filename': osp.join('to_anno', sensor_file),
                                'calibration': {
                                    'extrinsic_rotation': extrinsic_rotation,
                                    'extrinsic_translation': extrinsic_translation
                                },
                                'pose': {
                                    'rotation_to_map': r_sensor_to_map,
                                    'translation_to_map': t_sensor_to_map,
                                    'rotation_to_sensor': np.linalg.inv(sensor_to_map)[:3, :3],
                                    'translation_to_sensor': np.linalg.inv(sensor_to_map)[:3, 3].reshape((3,1))
                                } if len(pose_file) != 0 else None
                            }
                        else:
                            extrinsic_rotation = np.array(scene_info['calib']['camera_rotation'][sensor_name]).reshape((3,3))
                            extrinsic_translation = np.array(scene_info['calib']['camera_translation'][sensor_name]).reshape((3,1))
                            if len(pose_file) != 0:
                                r_sensor_to_map = pose_rota @ extrinsic_rotation
                                t_sensor_to_map = pose_rota @ extrinsic_translation + pose_trans
                                sensor_to_map = np.identity(4)
                                sensor_to_map[:3, :3] = r_sensor_to_map
                                sensor_to_map[:3, 3] = t_sensor_to_map.T
                            sensor_data = {
                                'sensor_name': sensor_name,
                                'sensor_type': sensor_type,
                                'timestamp': int(sensor_time),
                                'filename': osp.join('to_anno', sensor_file),
                                'calibration': {
                                    'intrinsic': np.array(scene_info['calib']['camera_intrinsic'][sensor_name]).reshape((3,3)),
                                    'extrinsic_rotation': extrinsic_rotation,
                                    'extrinsic_translation': extrinsic_translation,
                                },
                                'pose': {
                                    'rotation_to_map': r_sensor_to_map,
                                    'translation_to_map': t_sensor_to_map,
                                    'rotation_to_sensor': np.linalg.inv(sensor_to_map)[:3, :3],
                                    'translation_to_sensor': np.linalg.inv(sensor_to_map)[:3, 3].reshape((3,1))
                                } if len(pose_file) != 0 else None
                            }
                        sample_data['data'].append(sensor_data)
                    # 当前帧所有传感器信息添加到当前场景的samples信息中
                    scene_data['samples'].append(sample_data)
            # 保存当前场景离线数据
            os.makedirs(os.path.dirname(scene_data_npy_name), exist_ok=True)
            np.save(scene_data_npy_name, scene_data)
            # 当前场景信息添加到当前数据集信息中
            scenes_data['scenes'].append(scene_data)
        
        return scenes_data
