from os import path as osp
from importlib import import_module


class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config


    def _check_config(self, config_dict):
        '''
            check config for dataset
        '''
        # dataset path
        if not osp.exists(config_dict['data_dir']):
            raise ValueError('No data path')
        # dataset
        if not config_dict['dataset'] in ['Nuscenes', 'DDAD', 'KITTI', 'Neolix']:
            raise ValueError('No data %s'%config_dict['dataset'])
        # pose
        if config_dict['pose_mode'] not in ['pose', 'pred_pose']:
            raise ValueError('Wrong key-word for pose')


    def load_data(self):
            
        self._check_config(self.config)
        # 数据读取
        # Datasets = import_module('datasets.{}'.format(self.config['dataset']))
        Datasets = import_module(self.config['dataset'])
        if self.config['dataset'] == 'Neolix':
            dataroot = self.config['data_dir']
            pck_path = osp.join(self.config['data_dir'], 'bevdata_infos.pkl')
            dataset = Datasets.Datasets(dataroot,
                                        pck_path,
                                        self.config['scenes_classification']['special_scene'])
        else:
            print("not support dataset name: {}, please check config!".format(self.config['dataset']))
            exit(-1)
        dataset_data = dataset.get_data()

        pred_pose_dir = self.config['pred_pose_dir']
        is_absolute_pose = self.config['is_absolute_pose']
        ref_sensor_name = self.config['main_sensor']
        print('Pose Mode: {}'.format(self.config['pose_mode']))
        if self.config['pose_mode'] == 'pred_pose':
            if osp.exists(pred_pose_dir):
                dataset_data, is_success = dataset.get_pred_data(dataset_data, is_absolute_pose, pred_pose_dir, ref_sensor_name)
                if not is_success:
                    raise ValueError('Failed to load pred_pose from path: {}'.format(pred_pose_dir))
            else:
                raise ValueError('No path: {}'.format(pred_pose_dir))
        
        print("Finished to load dataset {}".format(self.config['dataset']))

        return dataset_data, dataroot