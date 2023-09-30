import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from os import path as osp
from importlib import import_module


class BaseDatasets(object):
    def __init__(self) -> None:
        pass


    def _get_pose_info(self, pose_path):
        '''
            get pred_pose_info from pred pose txt path
        '''
        with open(pose_path, 'r') as rf:
            pose_info_list = rf.readlines()
        pose_info = {}
        for pose_line in pose_info_list:
            pose_line_list = pose_line.rstrip().split(' ')
            lidar_name = pose_line_list[1]
            # timestamp = pose_line_list[1]
            pose_list = [float(i) for i in pose_line_list[2:9]]
            translation = np.array(pose_list[:3]).reshape((3, 1))
            rotation = R.from_quat(pose_list[3:]).as_matrix()
            pose_info[lidar_name] = np.hstack([rotation, translation])
        # print('pose_info')
        # print(pose_info)
        return pose_info


    def set_sample_pred_pose_none(self, sample_data):
        '''
        '''
        for sensor_data in sample_data['data']:
            sensor_data['pred_pose'] = None
        return sample_data


    def get_pred_data(self, dataset_data, is_absolute_pose, pred_pose_dir, ref_sensor_name):

        if len(os.listdir(pred_pose_dir)) == 0:
            return dataset_data, False
        for scene_data in dataset_data['scenes']:
            is_set_relative_to_absolute_pose = False
            scene_name = scene_data['scene_name']
            # pred_pose_path = osp.join(pred_pose_dir, '{}.txt'.format(scene_name))
            pred_pose_path = osp.join(pred_pose_dir, 'origin.txt')
            if not osp.exists(pred_pose_path):
                
                for sample_data in scene_data['samples']:
                    sample_data = self.set_sample_pred_pose_none(sample_data)
                continue
            pred_pose_info = self._get_pose_info(pred_pose_path)
            for sample_data in scene_data['samples']:
                ref_extrinsic_rotation = np.identity(3)
                ref_extrinsic_translation = np.zeros((3, 1))
                ref_rotation_to_map = np.identity(3)
                ref_translation_to_map = np.zeros((3, 1))
                for sensor_data in sample_data['data']:
                    if ref_sensor_name == sensor_data['sensor_name']:
                        ref_sensor_key = osp.basename(sensor_data['filename'])
                        pred_pose = pred_pose_info.get(osp.splitext(ref_sensor_key)[0], np.array([]))
                        # print(':::')
                        # print(pred_pose)
                        # print('osp.splitext(ref_sensor_key)[0]: ')
                        # print(osp.splitext(ref_sensor_key)[0])
                        if pred_pose.size == 0:
                            sample_data = self.set_sample_pred_pose_none(sample_data)
                            break
                        else:
                            if is_absolute_pose:
                                # print('33333333')
                                if not is_set_relative_to_absolute_pose:
                                    first_pred_pose = np.identity(4)
                                    first_pred_pose[:3,:3] = pred_pose[:,:3]
                                    first_pred_pose[:3,3] = pred_pose[:,3]
                                    reltative0_to_first_pred_pose = np.linalg.inv(first_pred_pose)
                                    is_set_relative_to_absolute_pose = True

                            else:
                                if sensor_data['pose'] == None:
                                    sample_data = self.set_sample_pred_pose_none(sample_data)
                                    break
                                elif not is_set_relative_to_absolute_pose:
                                    reltative0_to_first_pred_pose = np.identity(4)
                                    reltative0_to_first_pred_pose[:3,:3] = sensor_data['pose']['rotation_to_map'] @ \
                                                                            np.linalg.inv(pred_pose[:,:3])
                                    reltative0_to_first_pred_pose[:3,3] = (sensor_data['pose']['translation_to_map'] - \
                                                                        reltative0_to_first_pred_pose[:3,:3] @ \
                                                                        pred_pose[:,3].reshape((3,1))).T
                                    is_set_relative_to_absolute_pose = True

                        ref_pred_pose_rota = reltative0_to_first_pred_pose[:3,:3] @ pred_pose[:,:3]
                        ref_pred_pose_trans = reltative0_to_first_pred_pose[:3,3].reshape((3,1)) + \
                                            reltative0_to_first_pred_pose[:3,:3] @ \
                                            pred_pose[:,3].reshape((3,1))
                        ref_pred_pose_to_map = np.identity(4)
                        ref_pred_pose_to_map[:3, :3] = ref_pred_pose_rota
                        ref_pred_pose_to_map[:3, 3] = ref_pred_pose_trans.T
                        sensor_data['pred_pose'] = {
                            'rotation_to_map': ref_pred_pose_rota,
                            'translation_to_map': ref_pred_pose_trans,
                            'rotation_to_sensor': np.linalg.inv(ref_pred_pose_to_map)[:3, :3],
                            'translation_to_sensor': np.linalg.inv(ref_pred_pose_to_map)[:3, 3].reshape((3, 1))
                        }
                        ref_rotation_to_map = sensor_data['pred_pose']['rotation_to_map']
                        ref_translation_to_map = sensor_data['pred_pose']['translation_to_map']
                        ref_extrinsic_rotation = sensor_data['calibration']['extrinsic_rotation']
                        ref_extrinsic_translation = sensor_data['calibration']['extrinsic_translation']
                    for sensor_data in sample_data['data']:
                        if ref_sensor_name != sensor_data['sensor_name']:
                            sensor_pred_pose = {}
                            sensor_pred_pose['rotation_to_map'] = ref_rotation_to_map @ \
                                                           np.linalg.inv(
                                                               ref_extrinsic_rotation) @ \
                                                           sensor_data['calibration']['extrinsic_rotation']
                            sensor_pred_pose['translation_to_map'] = ref_translation_to_map + \
                                                              (
                                                                      ref_rotation_to_map @
                                                                      np.linalg.inv(ref_extrinsic_rotation) @
                                                                      (
                                                                              sensor_data['calibration']['extrinsic_translation'] -
                                                                              ref_extrinsic_translation
                                                                      )
                                                              )
                            pred_pose_to_map = np.identity(4)
                            pred_pose_to_map[:3, :3] = sensor_pred_pose['rotation_to_map']
                            pred_pose_to_map[:3, 3] = sensor_pred_pose['translation_to_map'].T
                            sensor_pred_pose['rotation_to_sensor'] = np.linalg.inv(pred_pose_to_map)[:3, :3]
                            sensor_pred_pose['translation_to_sensor'] = np.linalg.inv(pred_pose_to_map)[:3, 3].reshape((3, 1))
                            sensor_data['pred_pose'] = sensor_pred_pose
                    # print('sensor_pred_pose')
                    # print(sensor_pred_pose)

        return dataset_data, True