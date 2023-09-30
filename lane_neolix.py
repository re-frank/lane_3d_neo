import json
import os
import sys
import time
import copy
import glob
import cv2
import pprint
import functools
import scipy.special
import numpy as np
import open3d as o3d
import os.path as osp
from scipy import interpolate
from dataloader import DataLoader
from collections import defaultdict
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.interpolate import make_interp_spline
from numpy.polynomial.polynomial import Polynomial



def split_info_loader_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        split_file_path = func(*args, **kwargs)
        if not osp.isfile(split_file_path):
            split_list = []
        else:
            split_list = set(map(lambda x: x.strip(), open(split_file_path).readlines()))
            #print(split_list)
        return split_list
    return wrapper

class frame:
    line_2D = []
    def __init__(self, seq_id, frame_id):
        self.seq_id = seq_id
        self.frame_id = frame_id
        self.img_buf = []
        self.extrinsic_rotation = []
        self.extrinsic_translation = []
        self.pose_r = []
        self.pose_t = []
        self.intrinsic = []
        self.json_path = ''
        self.lanes = []
        self.line_2D = []
        self.lidar_timestamp, self.image_timestamp = self.extract_timestamps(frame_id)
        self.lanes_3D = []
    def pose_T(self):
        self.pose_T = np.eye(4)
        this_t = self.pose_t.ravel()
        self.pose_T[:3, :3] = self.pose_r
        self.pose_T[:3, 3] = this_t
        return self.pose_T

    def extract_timestamps(self, frame_id):
        parts = frame_id.split("_")
        if len(parts) == 2:
            lidar_timestamp = str(parts[0])
            image_timestamp = str(parts[1])
            return lidar_timestamp, image_timestamp
        else:
            raise ValueError("Invalid frame_id format: " + frame_id)

    def point_2D_to_lanes(self):
        if len(self.lanes) >= 2:
            for each_l in self.lanes:
                # 获得每条车道线的X和Y坐标
                x = [point[0] for point in each_l]
                y = [point[1] for point in each_l]

                # 使用 make_interp_spline 进行Bézier曲线拟合
                t = np.linspace(0, 1, len(x))  # 参数t，可以根据需要调整
                # spl = make_interp_spline(t, np.column_stack((x, y)), k=3, bc_type='not-a-knot')
                spl = make_interp_spline(t, np.column_stack((x, y)), k=3, bc_type='natural')

                
                # 计算拟合后的曲线上的点
                num_points_on_curve = 400  # 可根据需要调整
                t_new = np.linspace(0, 1, num_points_on_curve)
                points_on_curve = spl(t_new)
                points_on_curve_int = np.round(points_on_curve).astype(int)
                # 将曲线上的点添加到2D线列表
                self.line_2D.append(points_on_curve_int.tolist())

    def merge_lanes(self, new_frame_to_merge):
        start_time_merge = time.time()
        threshold_distance = 0.15
        print('one.points: ')

        if not self.lanes_3D:
            self.lanes_3D.extend(new_frame_to_merge)
            return

        merged_lanes = []
        for one in self.lanes_3D:
            print('one.points: ')
            print(one.points)
        # self.points_lidar_xyz = [line for line in self.points_lidar_xyz if not line.points]
        
        for new_lane in new_frame_to_merge:
            merged_lanes = []
            merged = False
            check = False
            for existing_lane in self.lanes_3D:
                for point2 in existing_lane.points:
                    if merged or check:
                        break
                    for point1 in new_lane.points:
                        # 计算当前点对之间的距离
                        if np.linalg.norm(np.array(point1) - np.array(point2)) < threshold_distance:
                            merged_lane = Lane(existing_lane.points + new_lane.points)
                            merged = True
                            check = True
                            break
                if not check:
                    merged_lanes.append(existing_lane)
                if merged and check:
                    merged_lanes.append(merged_lane)
                    check = False
            if not merged:
                merged_lanes.append(new_lane)  
            print('merged_lanes: ')
            print(merged_lanes)    
            print('len merged_lanes: ')
            print(len(merged_lanes))           
            self.lanes_3D.clear()
            self.lanes_3D.extend(merged_lanes)
        end_time_merge = time.time()
        elapsed_time = end_time_merge - start_time_merge
        print(f"merge代码运行时间：{elapsed_time:.4f} 秒")



    # def point_2D_to_lanes(self):
    #     if self.lanes:
    #         for each_l in self.lanes:
    #             eve_lane_pixels = []
    #             for i in range(len(each_l) - 1):
    #                 x1, y1 = int(each_l[i][0]), int(each_l[i][1])
    #                 x2, y2 = int(each_l[i + 1][0]), int(each_l[i + 1][1])
    #                 pixels_on_line = self.get_pixels_on_line(x1, y1, x2, y2)
    #                 eve_lane_pixels.extend(pixels_on_line)
    #             self.line_2D.append(eve_lane_pixels)

    def get_pixels_on_line(self, x1, y1, x2, y2):
        pixels_on_line = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            pixels_on_line.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return pixels_on_line

class Lane:
    def __init__(self, points):
        if points:
            start_time_init = time.time()
            self.points = points
            self.points_sorted_by_x = sorted(points, key=lambda point: point[0])
            self.points_sorted_by_y = sorted(points, key=lambda point: point[1])
            self.points_sorted_by_z = sorted(points, key=lambda point: point[2])
            self.two_p = []
            self.calculate_differences()
            self.points_2d = []
            end_time_init = time.time()
            elapsed_time = end_time_init - start_time_init
            print('init time: ')
            print(f"init代码运行时间：{elapsed_time:.4f} 秒")
            # self.points_2d = [[point[0], point[1], 0] for point in self.points]
            # self.update_points_2d()
        else:
            self.two_p = []
            self.points = []
            self.points_2d = []
            self.points_sorted_by_x = []
            self.points_sorted_by_y = []
            self.points_sorted_by_z = []
            self.sorted_points = []
            self.sort_method = None

    def calculate_differences(self):
        
        x_diff = self.points_sorted_by_x[-1][0] - self.points_sorted_by_x[0][0]
        y_diff = self.points_sorted_by_y[-1][1] - self.points_sorted_by_y[0][1]
        z_diff = self.points_sorted_by_y[-1][2] - self.points_sorted_by_y[0][2]
        if x_diff > y_diff and x_diff > z_diff:
            self.sorted_points = self.points_sorted_by_x
            self.sort_method = "x"
            self.two_p.append(self.points_sorted_by_x[0]) 
            self.two_p.append(self.points_sorted_by_x[-1]) 
        elif y_diff > x_diff and y_diff > z_diff:
            self.sorted_points = self.points_sorted_by_y
            self.sort_method = "y"
            self.two_p.append(self.points_sorted_by_y[0]) 
            self.two_p.append(self.points_sorted_by_y[-1]) 
        else:
            self.sorted_points = self.points_sorted_by_z
            self.sort_method = "z"
            self.two_p.append(self.points_sorted_by_z[0]) 
            self.two_p.append(self.points_sorted_by_z[-1]) 
        
    

    def get_sorted_points(self):
        return self.sorted_points

    def get_sort_method(self):
        return self.sort_method

        
    def interpolate(self, num_points=10000):
        start_time_berns = time.time()
        if not self.sorted_points:
            return []
        
        n = len(self.sorted_points) - 1
        
        def bernstein_poly(i, n, t):
            t_zero = np.where(t == 0, 1, t)
            t_one = np.where(t == 1, 1, t)
            
            # Check for invalid values before computation
            t_zero = np.where(np.isnan(t_zero) | np.isinf(t_zero), 1, t_zero)
            t_one = np.where(np.isnan(t_one) | np.isinf(t_one), 1, t_one)
            print('i n t t_zero: ')
            print(i)
            print(n)
            print(t)
            print(t_zero)
            result = scipy.special.comb(n, i) * (t ** i) * ((1 - t_zero) ** (n - i))
            return result

        
        t = np.linspace(0, 1, n+1)
        t_new = np.linspace(0, 1, num_points)

        x = [point[0] for point in self.sorted_points]
        y = [point[1] for point in self.sorted_points]
        z = [point[2] for point in self.sorted_points]

        x_interp = np.zeros(num_points)
        y_interp = np.zeros(num_points)
        z_interp = np.zeros(num_points)

        for i in range(n+1):
            x_interp += x[i] * bernstein_poly(i, n, t_new)
            y_interp += y[i] * bernstein_poly(i, n, t_new)
            z_interp += z[i] * bernstein_poly(i, n, t_new)

        self.dense_points = np.column_stack((x_interp, y_interp, z_interp))
        end_time_berns = time.time()
        elapsed_time_berns = end_time_berns - start_time_berns
        print('bernstein time: ')
        print(f"bernstein代码运行时间：{elapsed_time_berns:.4f} 秒")
        print('self.dense_points: ')
        print(self.dense_points)
        return self.dense_points


    def interpolate_2d(self, num_points=100):
        if not self.points_2d:
            return []

        x = np.array([point[0] for point in self.points_2d])
        y = np.array([point[1] for point in self.points_2d])

        # Create an Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack((x, y, np.zeros_like(x))))

        # Perform RANSAC-based line fitting
        _, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)

        # Extract inlier points
        inlier_indices = np.where(inliers)[0]
        inlier_points = pcd.select_by_index(inlier_indices)

        # Fit a polynomial curve to inlier points
        degree = 3
        x_inliers = np.array(inlier_points.points)[:, 0]
        y_inliers = np.array(inlier_points.points)[:, 1]
        poly_fit = np.polyfit(x_inliers, y_inliers, degree)

        # Generate interpolated points
        x_interp = np.linspace(min(x), max(x), num_points)
        y_interp = np.polyval(poly_fit, x_interp)

        dense_points_2D = np.column_stack((x_interp, y_interp, np.zeros_like(x_interp)))
        return dense_points_2D

    # def update_points_2d(self):
    #     if self.sort_method == 'x':
    #         x_set = set()
    #         for point in self.points:
    #             x_value = point[0]
    #             while x_value in x_set:
    #                 x_value += 1e-6  # Tiny offset to ensure uniqueness
    #             x_set.add(x_value)
    #             self.points_2d.append([x_value, point[1], 0])
    #     else:
    #         y_set = set()
    #         for point in self.points:
    #             y_value = point[1]
    #             while y_value in y_set:
    #                 y_value += 1e-6  # Tiny offset to ensure uniqueness
    #             y_set.add(y_value)
    #             self.points_2d.append([point[0], y_value, 0])
        
    # def get_2D_middle(self):
    #     if not self.points_2d:
    #         return None

    #     x_sum = 0
    #     y_sum = 0
    #     num_points = len(self.points_2d)

    #     for point in self.points_2d:
    #         x_sum += point[0]
    #         y_sum += point[1]

    #     x_avg = x_sum / num_points
    #     y_avg = y_sum / num_points

    #     return x_avg, y_avg
        # def interpolate(self, num_points=100):
    #     x = [point[0] for point in self.points]
    #     y = [point[1] for point in self.points]
    #     z = [point[2] for point in self.points]

    #     t = np.linspace(0, 1, len(x))
    #     t_new = np.linspace(0, 1, num_points)

    #     # x_interp = np.interp(t_new, t, x)
    #     # y_interp = np.interp(t_new, t, y)
    #     # z_interp = np.interp(t_new, t, z)

    #     if len(x) > 0:
    #         x_interp = np.interp(t_new, t, x)
    #     else:
    #         x_interp = np.zeros_like(t_new)  # 使用默认值或者其他操作
            
    #     if len(y) > 0:
    #         y_interp = np.interp(t_new, t, y)
    #     else:
    #         y_interp = np.zeros_like(t_new)  # 使用默认值或者其他操作

    #     if len(z) > 0:
    #         z_interp = np.interp(t_new, t, z)
    #     else:
    #         z_interp = np.zeros_like(t_new)  # 使用默认值或者其他操作

    #     dense_points = np.column_stack((x_interp, y_interp, z_interp))
    #     return dense_points
    # def interpolate(self, num_points=100):
    #     if len(self.points) < 2:
    #         # 如果点数小于2，直接返回原始点坐标
    #         return np.array(self.points)
    #     x = np.array([point[0] for point in self.points])
    #     y = np.array([point[1] for point in self.points])
    #     z = np.array([point[2] for point in self.points])

    #     t = np.linspace(0, 1, len(x))
    #     t_new = np.linspace(0, 1, num_points)

    #     cs_x = CubicSpline(t, x)
    #     cs_y = CubicSpline(t, y)
    #     cs_z = CubicSpline(t, z)

    #     x_interp = cs_x(t_new)
    #     y_interp = cs_y(t_new)
    #     z_interp = cs_z(t_new)

    #     dense_points = np.column_stack((x_interp, y_interp, z_interp))
    #     return dense_points
    # def interpolate(self, num_points=100):

    #     if not self.points:
    #         return []
    #     x = [point[0] for point in self.points]
    #     y = [point[1] for point in self.points]
    #     z = [point[2] for point in self.points]

    #     t = np.linspace(0, 1, len(x))
    #     t_new = np.linspace(0, 1, num_points)

    #     x_interp = np.interp(t_new, t, x)
    #     y_interp = np.interp(t_new, t, y)
    #     z_interp = np.interp(t_new, t, z)

    #     # Fit polynomial to the interpolated points
    #     order = 3  # Choose the polynomial order (cubic polynomial in this case)
    #     coeffs_x = np.polyfit(t_new, x_interp, order)
    #     coeffs_y = np.polyfit(t_new, y_interp, order)
    #     coeffs_z = np.polyfit(t_new, z_interp, order)

    #     # Evaluate the polynomial at the new points
    #     x_interp = np.polyval(coeffs_x, t_new)
    #     y_interp = np.polyval(coeffs_y, t_new)
    #     z_interp = np.polyval(coeffs_z, t_new)

    #     dense_points = np.column_stack((x_interp, y_interp, z_interp))
    #     return dense_points
class NEOLIX(object):
    """
    dataset structure:
    - data_root
        -ImageSets
            - train_split.txt
            - val_split.txt
            - test_split.txt
            - raw_split.txt
        - data
            - seq_id
                - cam01
                - cam03
                - ...
                -
    """
    #camera_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
    camera_names = ['front_left_1']
    camera_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']
    distortion = np.array([-0.3121, 0.1024, 0.00032953, -0.00039793, -0.0158])
    cam_to_velo = np.array([[-0.01084424, -0.01928796,  0.99975516, 0.35394928], [-0.99984618, -0.01357356, -0.0111071, 0.04280211], [0.01378447, -0.99972183, -0.0191378, -0.44652071], [0.0, 0.0, 0.0, 1.0]])
    all_covered_pixels = []
    points_lidar_xyz = []
    last_points_world = []
    god_lanes = []
    # new_points_xyz = np.empty((0, 3))
    def __init__(self, datasets, dataset_root):
        self.dataset_root = dataset_root
        self.datasets = datasets
        # self.data_root = osp.join(self.dataset_root, 'data')
        # self._collect_basic_infos()

    @property
    @split_info_loader_helper
    def train_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'train.txt')

    @property
    @split_info_loader_helper
    def val_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'val.txt')

    @property
    @split_info_loader_helper
    def test_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'test.txt')

    @property
    @split_info_loader_helper
    def raw_small_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_small.txt')

    @property
    @split_info_loader_helper
    def raw_medium_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_medium.txt')

    @property
    @split_info_loader_helper
    def raw_large_split_list(self):
        return osp.join(self.dataset_root, 'ImageSets', 'raw_large.txt')

    def _find_split_name(self, seq_id):
        print("self.train_split_list: ")
        print(self.train_split_list)
        if seq_id in self.raw_small_split_list:
            return 'raw_small'
        elif seq_id in self.raw_medium_split_list:
            return 'raw_medium'
        elif seq_id in self.raw_large_split_list:
            return 'raw_large'
        if seq_id in self.train_split_list:
            return 'train'
        if seq_id in self.test_split_list:
            return 'test'
        if seq_id in self.val_split_list:
            return 'val'
        print("sequence id {} corresponding to no split".format(seq_id))
        raise NotImplementedError

    def _collect_basic_infos(self):
        self.train_info = defaultdict(dict)
        self.val_info = defaultdict(dict)
        self.test_info = defaultdict(dict)
        self.raw_small_info = defaultdict(dict)
        self.raw_medium_info = defaultdict(dict)
        self.raw_large_info = defaultdict(dict)
        for attr in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']:
            if getattr(self, '{}_split_list'.format(attr)) is not None:
                split_list = getattr(self, '{}_split_list'.format(attr))
                # print("attr : " )
                # print(attr)
                # print("split_list : " )
                # print(split_list)
                info_dict = getattr(self, '{}_info'.format(attr))
                # print("info_dict : " )
                # print(info_dict)
                for seq in split_list:
                    #anno_file_path = osp.join(self.data_root, seq, '{}.json'.format(seq))
                    # seq = seq.lstrip('/')
                    # #print(seq)
                    # anno_file_path = osp.join(self.data_root,osp.dirname(osp.dirname(seq)), '{}.json'.format(osp.dirname(osp.dirname(seq))))
                    for cam_name in self.__class__.camera_names:
                        lane_frame_path = osp.join(self.data_root, seq, '{}json'.format(seq), cam_name)
                        for filename in os.listdir(lane_frame_path):
                            frame_list = list()
                            if filename.endswith('.json'):
                                frame_id = osp.splitext(filename)[0]
                                frame_list.append(frame_id)
                            # if not osp.isfile(anno_file_path):
                            #     print("no annotation file for sequence {}".format(seq))
                            #     raise FileNotFoundError
                                anno_file = json.load(open(lane_frame_path+ '/'+ filename, 'r'))
                                # print('anno_file: ')
                                # print(anno_file)
                                lane_num = anno_file['lane_num']
                                lanes = anno_file['lanes']
                                info_dict[seq][frame_id] = {} 
                                info_dict[seq][frame_id][cam_name] = dict()
                                info_dict[seq][frame_id][cam_name]['lane_num'] = lane_num
                                info_dict[seq][frame_id][cam_name]['lanes'] = lanes
                                # 提取 'calibration' 数据并存储到info_dict中
                                calibration_data = anno_file['calibration']
                                info_dict[seq][frame_id][cam_name]['calibration'] = calibration_data
                                # print('info_dict ' + frame_id + ' lane_num: ')
                                # print(info_dict[seq][frame_id]['lane_num'])
                                # time.sleep(9000000)

                        
                    pose_seq_path = osp.join(self.data_root, seq, '{}.json'.format(seq))    
                    if not osp.isfile(pose_seq_path):
                        print("no annotation file for sequence {}".format(seq))
                        raise FileNotFoundError
                    seq_file = json.load(open(pose_seq_path, 'r'))
                    for frame_pose in seq_file['frames']:
                        # print('11')
                        # frame_list.append(str(frame_pose['frame_id']))
                        frame_id = str(frame_pose['frame_id'])
                        # print('22')
                        if frame_id not in info_dict[seq]:
                            continue
                        info_dict[seq][frame_id]['pose'] = frame_pose['pose']
                        # print('info_dict[seq][frame_id][pose]:' )
                        # print(info_dict[seq][frame_id])
    def lidar2world(self, pose):
        quat_x, quat_y, quat_z, quat_w, trans_x, trans_y, trans_z = pose
        quat = np.array([quat_x, quat_y, quat_z, quat_w])
        # 转换为旋转矩阵
        rotation = Rotation.from_quat(quat)
        rotation_matrix = rotation.as_matrix()
        # 平移向量
        translation_vector = np.array([trans_x, trans_y, trans_z])
        # 对点云进行坐标变换
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        return rotation_matrix, translation_vector, transformation_matrix

    def merge_points(self, new_lane_to_merge, exist_lanes):
        start_time_merge = time.time()
        threshold_distance = 0.15
        print('one.points: ')
        print(new_lane_to_merge.points)
        # new_lane_to_merge.points = np.vstack(new_lane_to_merge.points)
        print('new_lane_to_merge: ')
        print(new_lane_to_merge.points)
        if not exist_lanes:
            exist_lanes.append(new_lane_to_merge)
            return
        merged_lanes = []
        for one in exist_lanes:
            print('one.points: ')
            print(one.points)
        # self.points_lidar_xyz = [line for line in self.points_lidar_xyz if not line.points]
        merged = False
        check = False
        for existing_lane in exist_lanes:
            for point2 in existing_lane.points:
                if merged or check:
                    break
                for point1 in new_lane_to_merge.points:
                    # 计算当前点对之间的距离
                    if np.linalg.norm(point1 - point2) < threshold_distance:
                        merged_lane = Lane(existing_lane.points + new_lane_to_merge.points)
                        merged = True
                        check = True
                        break
            if not check:
                print('888')
                merged_lanes.append(existing_lane)
            if merged and check:
                print('999')
                merged_lanes.append(merged_lane)
                check = False
        if not merged:
            print('777')
            merged_lanes.append(new_lane_to_merge)    
        print('len merged_lanes: ')
        print(len(merged_lanes))           
        self.points_lidar_xyz.clear()
        self.points_lidar_xyz.extend(merged_lanes)
        end_time_merge = time.time()
        elapsed_time = end_time_merge - start_time_merge
        print(f"merge代码运行时间：{elapsed_time:.4f} 秒")

    def get_frame_anno(self, seq_id, frame_id):
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        if 'annos' in frame_info:
            return frame_info['annos']
        return None

    def load_point_cloud(self, one_frame):
        #bin_path = osp.join(self.data_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
        get_flag = False
        bin_path = ''
        for record in self.datasets['scenes']:
            for frame in record['samples']:
                for sensor in frame['data']: 
                    if (sensor['sensor_name'] == 'top_lidar') and (os.path.basename(sensor['filename']) == (one_frame.lidar_timestamp + '.pcd')):
                        bin_path = osp.join('/home/duanqingchuan/lane_3d/', sensor['filename'])
                        get_flag = True
                        break
                if get_flag == True:
                    break 
        print(get_flag)
        # with open(bin_path, 'rb') as file:
        #     data = file.read()
        #     data_count = len(data) // 4  # 假设每个数据项占用4字节，这里简单地计算数据数量
        #     print("Number of data points in the file:", data_count)
        # points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
        pcd = o3d.io.read_point_cloud(bin_path)
        # 获取点云数据
        points = np.asarray(pcd.points)
        # points = np.fromfile(bin_path, dtype=np.float32)
        print('points: ')
        print(points)
        return points

    def load_image(self, seq_id, frame_id, cam_name):
        cam_path = osp.join(self.dataset_root, seq_id, cam_name, '{}.jpg'.format(frame_id))
        img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
        height, width, channels = img_buf.shape
        # print('img_shape: ' + ' height ' + str(height) + ' width ' + str(width) +' channels ' + str(channels))
        return img_buf

    def undistort_image(self, seq_id, frame_id):
        img_list = []
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        for cam_name in self.__class__.camera_names:
            img_buf = self.load_image(seq_id, frame_id, cam_name)
            cam_calib = frame_info['calib'][cam_name]
            h, w = img_buf.shape[:2]
            cv2.getOptimalNewCameraMatrix(cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          (w, h), alpha=0.0, newImgSize=(w, h))
            img_list.append(cv2.undistort(img_buf, cam_calib['cam_intrinsic'],
                                          cam_calib['distortion'],
                                          newCameraMatrix=cam_calib['cam_intrinsic']))
        return img_list

    def undistort_image_v2(self, seq_id, frame_id, cam_name):
        img_list = []
        new_cam_intrinsic_dict = dict()
        split_name = self._find_split_name(seq_id)
        # frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        frame_info = self.datasets['scenes']['samples']
        for cam_name in self.__class__.camera_names:
            img_buf = self.load_image(seq_id, frame_id, cam_name)
            # cam_calib = frame_info['calibration'][cam_name]
            cam_calib = frame_info[cam_name]['calibration']
            print('frame_info[calibration]')
            print(frame_info[cam_name]['calibration'])
            calibration_matrix = np.array(frame_info[cam_name]['calibration'], dtype=np.float32)[:3, :3]
            img_list.append()
        return img_list, new_cam_intrinsic_dict

  
    def project_lanes_to_image(self, this_frame):
        # 读取json文件中的2D标注信息
        this_frame.json_path = osp.join(self.dataset_root,this_frame.seq_id, '{}_anno'.format(self.camera_names[0]), '{}.lines.json'.format(this_frame.frame_id))
        anno_file = json.load(open(this_frame.json_path, 'r'))
        this_frame.lanes = anno_file["annotations"]["lanes"]

        print('this_frame.lanes: ')
        print(this_frame.lanes)
        # 将标注点拟合成线
        this_frame.point_2D_to_lanes()
        # 将点和线都画到图上
        for each_l in this_frame.lanes:
            for x, y in each_l:
                    x = int(x)
                    y = int(y)
                    cv2.circle(this_frame.img_buf, (x, y), 2, (100, 0, 255), -1)
        for each_line in this_frame.line_2D:
                for x, y in each_line:
                    x = int(x)
                    y = int(y)
                    cv2.circle(this_frame.img_buf, (x, y), 0, (255, 0, 0), -1)


    def project_lidar_to_image(self, frame_list, this_frame):
        print('this_frame.frame_id')
        print(this_frame.frame_id)
        this_index = frame_list.index(this_frame)
        # 一. 点云到当前帧图像上的投影
         # 创建一个新的空列表用于存储新的点云投影
         # 获得当前帧点云
        this_points = self.load_point_cloud(this_frame) 
         # 获得lidar与相机的外参
        cam_2_velo = self.cam_to_velo
         # 获得相机内参
        print('this_frame.intrinsic: ')
        print(this_frame.intrinsic)
        cam_intri = np.hstack([this_frame.intrinsic, np.zeros((3, 1), dtype=np.float32)])
        
        if len(this_frame.pose_r) == 0:
            return
         # 获得该帧位姿，即lidar到世界系的变换
        this_r = this_frame.pose_r
        this_t = this_frame.pose_t.ravel() # 多维展平为一维
        this_T = this_frame.pose_T
         # 获得lidar点云的世界坐标
        points_lidar = this_points[:, :3]
        point_xyz = this_r.dot(points_lidar.T).T + this_t
        print('point_xyz: ')
        print(point_xyz)
        print('len point_xyz: ')
        print(len(point_xyz))
         # 扩展为齐次坐标
        points_homo = np.hstack(
            [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
         # 获得相机到世界坐标系的变换
        cam_2_world = np.dot(this_T, cam_2_velo)
        
         # 获得点云在相机系下的坐标
        points_cam = np.dot(points_homo, np.linalg.inv(cam_2_world).T)
        mask = (points_cam[:, 2] > 0) & ((points_cam[:, 2] < 23)) # 距离控制
        points_cam = points_cam[mask]
         # 对5m-30m的未来帧点云每隔5帧选取一次向当前帧投影
        if this_frame is not frame_list[-1]:
            nex_index = this_index + 1
            nex_frame = frame_list[nex_index]
            while (nex_index + 5 < len(frame_list)) and (calculate_distance_between_frames(this_frame.pose_t, nex_frame.pose_t) < 5):
                print('instance < 2')
                print(calculate_distance_between_frames(this_frame.pose_t, nex_frame.pose_t))
                print('nex_index')
                print(nex_index)
                nex_index = nex_index + 5
                nex_frame = frame_list[nex_index]
            key_frame = this_frame
            len_frame_list = len(frame_list)
            dis = calculate_distance_between_frames(this_frame.pose_t, nex_frame.pose_t)
            while (nex_index + 5 < len(frame_list)) and (calculate_distance_between_frames(this_frame.pose_t, nex_frame.pose_t) < 30):
                if (len(nex_frame.pose_r) == 0) or (calculate_distance_between_frames(key_frame.pose_t, nex_frame.pose_t) < 5):
                    nex_index = nex_index + 5
                    nex_frame = frame_list[nex_index]
                    continue
                nex_points = self.load_point_cloud(nex_frame)
                nex_r = nex_frame.pose_r
                nex_t = nex_frame.pose_t.ravel()
                nex_T = nex_frame.pose_T
                nex_index = nex_index + 5
                nex_frame = frame_list[nex_index]
                nex_points_lidar = nex_points[:, :3]
                nex_point_xyz = nex_r.dot(nex_points_lidar.T).T + nex_t
                nex_points_homo = np.hstack(
                    [nex_point_xyz, np.ones(nex_point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
                # nex_cam_2_world = np.dot(nex_T, cam_2_velo)
                # 先用这一帧的图像选点，再往上一帧图像投影
                # nex_points_cam_temp = np.dot(nex_points_homo, np.linalg.inv(nex_cam_2_world).T)
                nex_points_cam = np.dot(nex_points_homo, np.linalg.inv(cam_2_world).T)
                # nex_mask = (nex_points_cam_temp[:, 2] < 0) & (nex_points_cam[:, 2] > 0)
                nex_mask = nex_points_cam[:, 2] > 0
                nex_points_cam = nex_points_cam[nex_mask]
                # point_xyz = np.concatenate((point_xyz, nex_point_xyz))
                point_xyz = np.concatenate((point_xyz, nex_point_xyz))
                print('len point_xyz: ')
                print(len(point_xyz))
                points_cam = np.concatenate((points_cam, nex_points_cam))
                mask = np.concatenate((mask, nex_mask))
                key_frame = nex_frame
        

        print('len point_xyz: ')
        print(len(point_xyz))
         # 将相机系下的点投到成像平面
        points_img = np.dot(points_cam, cam_intri.T)
        points_img = points_img / points_img[:, [2]]
        points_img = np.round(points_img).astype(int)
        print('points_img: ')
        print(points_img)
        print('points_img len')
        print(len(points_img))
        
         # 画图
        for point in points_img:
            try:
                cv2.circle(this_frame.img_buf, (int(point[0]), int(point[1])), 2, color=(0, 0, 255), thickness=-1)
            except:
                print('warning!!!')
                print(int(point[0]), int(point[1]))

        # 二. 对于all_covered_pixels中的每个像素点，找到最近的点云投影并添加到new_points_img列表中
        new_points_img = [] # 2D车道线标注覆盖的点云投影
        new_points_lidar_all_line = []
         # 对当前帧图像中的车道线2D标注的拟合线逐像素遍历，找到近似重合的点云投影点
        for each_line in this_frame.line_2D:
            new_points_lidar_each_line = []
            new_points_img_each_line = []
            # 为不同位置的车道线赋不同的距离阈值
            x_sum = 0
            num_points = len(each_line)
            for point in each_line:
                x_sum += point[0]
            x_avg = x_sum / num_points
            print('x_avg: ')
            print(x_avg)
            if x_avg < 1000 or x_avg > 3000:
                ori_min_distance = 2
            else:
                ori_min_distance = 2

            start_time_near = time.time()

            # each_arr = np.array(each_line)

            image_size = (2160, 3840)
            img_mask = np.zeros(image_size, dtype=np.uint8)
            for center in each_line:
                x, y = center
                #！#
                # 在mask上绘制圆形区域
                cv2.circle(img_mask, (int(x), int(y)), ori_min_distance, 1, -1)  # -1 表示填充
                cv2.circle(this_frame.img_buf, (int(x), int(y)), ori_min_distance, 1, -1)
            # 遍历像素点
            points_within_mask = np.transpose(np.where(img_mask == 1))
            points_within_mask = points_within_mask[:, [1, 0]]
            # points_within_mask = np.where(img_mask == 1)
            points_img_xy = points_img[:, :2]
            # print('points_within_mask')
            # for point in points_within_mask:
            #     print(point)
            # print('points_img_xy')
            # for point in points_img_xy:
            #     print(point)
            # has_duplicates = len(points_img_xy) != len(np.unique(points_img_xy, axis=0))
            # mask_1d = points_within_mask.flatten()
            mask_1d_product = points_within_mask[:,1]*3840+points_within_mask[:, 0]
            # points_img_1d = points_img_xy.flatten()
            # points_img_1d_product = np.prod(points_img_1d.reshape(-1, 2), axis=1)
            points_img_1d_product = points_img_xy[:,1]*3840+points_img_xy[:, 0]
            same_p,line_index,lidar_index = np.intersect1d(mask_1d_product.tolist(), points_img_1d_product.tolist(), return_indices = True)
            if len(lidar_index) == 0:
                continue
            for index in lidar_index:
                x,y = points_img_xy[index]
                if x < 0 or x > 3840 or y < 0 or y > 2160:
                    continue
                new_points_img_each_line.append(points_img[index])
                corresponding_point_xyz = point_xyz[mask][index]
                new_points_lidar_each_line.append(corresponding_point_xyz)


            # for x, y in each_line:
            #     # 初始化最小距离和对应的最近点云投影
            #     nearest_point = None
            #     nearest_point_index = None
            #     min_distance = ori_min_distance
            #     # 寻找最近的点云投影
            #     for idx, point in enumerate(points_img):
            #         distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            #         if distance < min_distance:
            #             min_distance = distance
            #             nearest_point = point
            #             nearest_point_index = idx

            #     # 添加最近点云投影到新列表中
            #     if nearest_point is not None:
            #         # 储存二维点
            #         new_points_img_each_line.append(nearest_point)
            #         # 对应的三维点云添加到新列表中
            #         corresponding_point_xyz = point_xyz[mask][nearest_point_index]
            #         new_points_lidar_each_line.append(corresponding_point_xyz)
            #         # new_points_xyz[idx] = corresponding_point_xyz


            if len(new_points_img_each_line) > 3:
                new_points_img.extend(new_points_img_each_line)
                print("new_points_lidar_each_line: ")
                print(new_points_lidar_each_line)
                # 实例化车道线，并合并入已有车道线
                new_lane = Lane(new_points_lidar_each_line)
                self.merge_points(new_lane, self.points_lidar_xyz)
                # self.merge_points(new_lane, self.god_lanes)
            end_time_near = time.time()
            elapsed_time_near = end_time_near - start_time_near
            print('near time: ')
            print(f"投影代码运行时间：{elapsed_time_near:.4f} 秒")
        # 对车道线点云进行下采样
        for each_l in self.points_lidar_xyz:
            print("each_l.points: ")
            print(each_l.points)
            each_l.points = self.voxel_filter(each_l.points)
            # while len(each_l.points) > 600 :
            #     print("filter do once!")
            #     each_l.points = self.voxel_filter(each_l.points)
            new_points_lidar_all_line.append(each_l.points)
            print("each_l.points: ")
            print(each_l.points)
        # 三. 将车道线点云拟合并投影到图像中
        pcd_line = []
        pcd_2d_line = []
        output_lanes = []
        flag_past = False
        color_size = 0
        lines_size = len(self.points_lidar_xyz)
        # points_lidar_xyz_arr = np.array(self.points_lidar_xyz[3:])
        # for 
        # points_temp_lidar = np.hstack(
        #         [self.points_lidar_xyz.points, np.ones(self.points_lidar_xyz.points.shape[0], dtype=np.float32).reshape((-1, 1))])
        # points_temp_cam_homo = np.dot( np.linalg.inv(this_frame.pose_T), points_temp_lidar)
        # lidar_mask = (points_temp_cam_homo[:, 1] > -10) & (points_temp_cam_homo[:, 1] < 40)
        # self.points_lidar_xyz = self.points_lidar_xyz[lidar_mask]
        for each_line in self.points_lidar_xyz:
            if len(each_line.points) == 0:
                continue
            arr_points = np.array(each_line.points)
            points_temp_lidar = np.hstack(
                    [arr_points, np.ones(arr_points.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_temp_cam_homo = np.dot( points_temp_lidar, np.linalg.inv(cam_2_world).T)
            cam_mask = (points_temp_cam_homo[:, 2] > -5) & (points_temp_cam_homo[:, 2] < 25)
            arr_points = arr_points[cam_mask]  
            each_line.points = arr_points.tolist()
            
        self.points_lidar_xyz = [each_line for each_line in self.points_lidar_xyz if any(each_line.points)]
        this_frame.lanes_3D = self.points_lidar_xyz.copy()
        output_lanes = self.points_lidar_xyz.copy()

        if this_frame is not frame_list[0]:
            last_index = this_index - 1
            last_frame = frame_list[last_index]
            while (last_index - 3 >= 0) and (calculate_distance_between_frames(this_frame.pose_t, last_frame.pose_t) < 5):
                last_index = last_index - 3
                last_frame = frame_list[last_index]
            print('calculate_distance_between_frames(this_frame.pose_t, last_frame.pose_t) = ')
            print(calculate_distance_between_frames(this_frame.pose_t, last_frame.pose_t))
            if (last_index >= 0) and (calculate_distance_between_frames(this_frame.pose_t, last_frame.pose_t) > 5):
                print('this_frame.lanes_3D len: ')
                print(len(this_frame.lanes_3D))
                print('last_frame.lanes_3D len: ')
                print(len(last_frame.lanes_3D))
                print('last_frame.frame_id: ')
                print(last_frame.frame_id)
                # last_frame.lanes_3D.extend(this_frame.lanes_3D)
                last_frame.merge_lanes(this_frame.lanes_3D)
                output_lanes = last_frame.lanes_3D
                flag_past = True
        #     else:
        #         this_frame.lanes_3D = self.points_lidar_xyz
        # else:
        #     this_frame.lanes_3D = self.points_lidar_xyz
          
        for each_line in output_lanes:   
            #### 将车道线点云拟合后投影回图像
            temp_points = np.array(each_line.interpolate(10000)) # 拟合结果
            lidar_points = temp_points.reshape(-1, 3)
            lidar_points_homo = np.hstack(
                [lidar_points, np.ones(lidar_points.shape[0], dtype=np.float32).reshape((-1, 1))])
            # points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            lidar_points_cam = np.dot(lidar_points_homo, np.linalg.inv(cam_2_world).T)
            # points_cam = np.dot(points_homo, cam_2_world)
            lidar_mask = lidar_points_cam[:, 2] > 0
            lidar_points_cam = lidar_points_cam[lidar_mask]
            lidar_points_img = np.dot(lidar_points_cam, cam_intri.T)
            lidar_points_img = lidar_points_img / lidar_points_img[:, [2]]
            #！#
            for point in lidar_points_img:
                try:
                    cv2.circle(this_frame.img_buf, (int(point[0]), int(point[1])), 2, color=(255, 20, 255), thickness=-1)
                except:
                    print('warning!!!')
                    print(int(point[0]), int(point[1]))

            # 分别为2D和3D的点云及拟合直线创建点云对象并将新的三维点云坐标赋值给点云对象
            pcd_one_line = o3d.geometry.PointCloud()
            pcd_one_points = o3d.geometry.PointCloud()
            pcd_one_points.points = o3d.utility.Vector3dVector(each_line.interpolate(10000))
            pcd_one_line.points = o3d.utility.Vector3dVector(each_line.points)

            # 设置新的点云颜色
            pcd_one_line.paint_uniform_color([0.5, color_size, color_size])
            pcd_one_points.paint_uniform_color([color_size, 0.5, color_size])

            color_size = color_size + 1 / lines_size

            pcd_line.append(pcd_one_line)
            pcd_line.append(pcd_one_points)
        #！#
        for point in new_points_img:
            cv2.circle(this_frame.img_buf, (int(point[0]), int(point[1])), 3, color=(255, 0, 0), thickness=-1)
        print('pcd_line.shape: ')
        print(len(pcd_line))
        print(pcd_line)

        # 将两个点云对象合并
        pcd_combined = o3d.geometry.PointCloud()
        lidar_pose = o3d.geometry.PointCloud()
        
        i = 0
        print('pcd_line len: ')
        print(len(pcd_line))
        for pcd_one in pcd_line:
            pcd_combined = pcd_one + pcd_combined
            # o3d.io.write_point_cloud(f"images/combined_points_{i}.pcd", pcd_one)
            i = i + 1
        # pcd_combined = pcd_combined + lidar_pose
        # 输出为pcd文件
        # o3d.io.write_point_cloud(f"images/combined_points_BEV_{frame_id}.pcd", pcd_combined_2d)
        pcd_point_xyz = o3d.geometry.PointCloud()
        pcd_point_xyz.points = o3d.utility.Vector3dVector(point_xyz)
        pcd_point_xyz.paint_uniform_color([0, 1, 0])

        if flag_past == True:
            pose_last = []
            pose_last.append(this_frame.pose_t.ravel().tolist())
            lidar_pose.points = o3d.utility.Vector3dVector(pose_last)
            lidar_pose.paint_uniform_color([0.5, 0.5, 0.5])
            pcd_combined = pcd_combined + lidar_pose
            o3d.io.write_point_cloud(f"result_100/lidar_points_{last_frame.frame_id}_new_3.pcd", pcd_point_xyz)
            o3d.io.write_point_cloud(f"result_100/combined_points_{last_frame.frame_id}_new_6.pcd", pcd_combined)
        else:
            pose_now = []
            pose_now.append(this_frame.pose_t.ravel().tolist())
            lidar_pose.points = o3d.utility.Vector3dVector(pose_now)
            lidar_pose.paint_uniform_color([0.5, 0.5, 0.5])
            pcd_combined = pcd_combined + lidar_pose
            o3d.io.write_point_cloud(f"result_100/lidar_points_{this_frame.frame_id}.pcd", pcd_point_xyz)
            o3d.io.write_point_cloud(f"result_100/combined_points_3_{this_frame.frame_id}.pcd", pcd_combined)
        
        print('last_points_world: ')
        print(self.last_points_world)
        if len(self.last_points_world) > 0:
            last_points = self.last_points_world.reshape(-1, 4)
            # last_points_homo = np.hstack(
            #     [last_points, np.ones(last_points.shape[0], dtype=np.float32).reshape((-1, 1))])
            # points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            last_points_cam = np.dot(last_points, np.linalg.inv(cam_2_world).T)
            # points_cam = np.dot(points_homo, cam_2_world)
            print('last_points_cam: ')
            print(last_points_cam)
            last_mask = last_points_cam[:, 2] > 0
            print('last_points_cam[:, 2]: ')
            print(last_points_cam[:, 2])
            # # print('points_cam[mask]: ')
            # # print(mask)
            last_points_cam = last_points_cam[last_mask]
            last_points_img = np.dot(last_points_cam, cam_intri.T)
            last_points_img = last_points_img / last_points_img[:, [2]]
            print('last_points_img: ')
            print(last_points_img)
            #！#
            for point in last_points_img:
                cv2.circle(this_frame.img_buf, (int(point[0]), int(point[1])), 3, color=(0, 255, 200), thickness=-1)

        


        test = []
        for test_line in new_points_lidar_all_line:
            # test.extend(test_line)
            test.extend(test_line)
        self.last_points_world = np.array(test)
        self.last_points_world = np.hstack(
                [self.last_points_world, np.ones(self.last_points_world.shape[0], dtype=np.float32).reshape((-1, 1))])


    def pcd_to_bin(pcd_file, bin_file):
        # 读取PCD文件
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # 获取点云数据
        points = pcd.points

        # 将点云数据保存为二进制文件
        points_np = points.numpy().astype('float32')
        points_np.tofile(bin_file)

   
    def frame_concat(self, seq_id, frame_id, concat_cnt=0):
        """
        return new points coordinates according to pose info
        :param seq_id:
        :param frame_id:
        :return:
        """
        split_name = self._find_split_name(seq_id)
        seq_info = getattr(self, '{}_info'.format(split_name))[seq_id]
        start_idx = seq_info['frame_list'].index(frame_id)
        points_list = []
        translation_r = None
        try:
            for i in range(start_idx, start_idx + concat_cnt + 1):
                current_frame_id = seq_info['frame_list'][i]
                frame_info = seq_info[current_frame_id]
                transform_data = frame_info['pose']
    
                points = self.load_point_cloud(seq_id, current_frame_id)
                points_xyz = points[:, :3]
    
                rotation = Rotation.from_quat(transform_data[:4]).as_matrix()
                translation = np.array(transform_data[4:]).transpose()
                points_xyz = np.dot(points_xyz, rotation.T)
                points_xyz = points_xyz + translation
                if i == start_idx:
                    translation_r = translation
                points_xyz = points_xyz - translation_r
                points_list.append(np.hstack([points_xyz, points[:, 3:]]))
        except ValueError:
            print('warning: part of the frames have no available pose information, return first frame point instead')
            points = self.load_point_cloud(seq_id, seq_info['frame_list'][start_idx])
            points_list.append(points)
            return points_list
        return points_list

    def smooth_line_thickness(self, line_thicknesses):
        if len(line_thicknesses) < 4:  # Adjust the threshold as needed
            return line_thicknesses
        # 使用插值方法对线宽进行平滑
        x = np.arange(len(line_thicknesses))
        # f = interpolate.interp1d(x, line_thicknesses, kind='cubic')
        f = interpolate.interp1d(x, line_thicknesses, kind='cubic', fill_value="extrapolate")
        smooth_x = np.linspace(0, len(line_thicknesses) - 1, num=len(line_thicknesses) * 10)
        smooth_line_thicknesses = f(smooth_x)
        return smooth_line_thicknesses

    def load_frame(self, frames):
        first_flag = True
        frames_to_remove = []
        for idx, this_frame in enumerate(frames):
            get_flag_l = False
            get_flag_f = False
            for cam_no, cam_name in enumerate(self.__class__.camera_names):
                for record in self.datasets['scenes']:
                    for frame in record['samples']:
                        for sensor in frame['data']:
                            if (sensor['sensor_name'] == 'top_lidar') and (os.path.basename(sensor['filename']) == (this_frame.lidar_timestamp + '.pcd')):
                                if sensor['pred_pose'] == None:
                                    if first_flag == True:
                                        frames_to_remove.append(idx)
                                    get_flag_l = True
                                else:
                                    first_flag = False
                                    this_frame.pose_r = sensor['pred_pose']['rotation_to_map']
                                    this_frame.pose_t = sensor['pred_pose']['translation_to_map']
                                    this_frame.pose_T()
                                    get_flag_l = True
                            if (sensor['sensor_name'] == 'front_left_1') and (os.path.basename(sensor['filename']) == this_frame.frame_id + '.jpg'):
                                file_name = os.path.splitext(os.path.basename(sensor['filename']))[0]
                                this_frame.extrinsic_rotation = sensor['calibration']['extrinsic_rotation']
                                this_frame.extrinsic_translation = sensor['calibration']['extrinsic_translation']
                                this_frame.intrinsic = sensor['calibration']['intrinsic']
                                img_buf = self.load_image(this_frame.seq_id, file_name, cam_name)
                                this_frame.img_buf = img_buf
                                get_flag_f = True
                            if get_flag_l and get_flag_f:
                                break
                        if get_flag_l and get_flag_f:
                            break
        for idx in reversed(frames_to_remove):
            del frames[idx]

    def voxel_filter(self, point_cloud, voxel_size=0.5):

        print(len(point_cloud))
        # 创建一个 Open3D 的点云对象
        pcd = o3d.geometry.PointCloud()
        # 将输入的点云数据转换为 NumPy 数组
        np_point_cloud = np.array(point_cloud, dtype=np.float32)
        # 设置点云数据
        pcd.points = o3d.utility.Vector3dVector(np_point_cloud)
        # 使用 voxel_down_sample 方法进行体素下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # 将下采样后的点云数据转换为与输入格式相同的数据
        filtered_point_cloud = [list(point) for point in np.asarray(downsampled_pcd.points)]
        print(len(filtered_point_cloud))
        return filtered_point_cloud







def input_seq( num, start_id, folder_dir, sensors):
    # jpg_files = glob.glob(os.path.join(folder_dir, '*.jpg'))
    jpg_files = []
    items = sorted(os.listdir(folder_dir))
    # flag = False  # 用于标记是否已找到start_id文件夹
    # remaining_num = num # 剩余需要获取的图片数量

    for item in items:
        for sensor in sensors:
            item_path = os.path.join(folder_dir, item, sensor)

            # 检查项目是否是一个文件夹
            if os.path.isdir(item_path):
                record_id = item
                files_in_folder = os.listdir(item_path)
                jpg_file = sorted([file for file in files_in_folder if file.endswith('.jpg')])
                jpg_files.extend([(record_id, file) for file in jpg_file])

    print('jpg_files len: ')
    print(len(jpg_files))
    count = 0
    frames = []
    flag = False
    for record_id, file in jpg_files:
        # 提取文件名（不包括路径和扩展名）
        file_name = os.path.splitext(os.path.basename(file))[0]
        # 如果文件名包含目标名称
        if start_id == file_name:
            flag = True
        if flag and count < num:
            frames.append((record_id, file_name))
            count += 1
            if count >= num:
                break
    return frames

# def input_seq(num, start_id, folder_dir, sensors):
#     jpg_files = []
#     items = sorted(os.listdir(folder_dir))
#     flag = False  # 用于标记是否已找到start_id文件夹
#     remaining_num = num  # 剩余需要获取的图片数量

#     for item in items:
#         for sensor in sensors:
#             item_path = os.path.join(folder_dir, item, sensor)

#             # 检查项目是否是一个文件夹
#             if os.path.isdir(item_path):
#                 record_id = item
#                 files_in_folder = os.listdir(item_path)
#                 jpg_file = sorted([file for file in files_in_folder if file.endswith('.jpg')])

#                 if flag:
#                     # 如果已找到start_id文件夹，直接添加所有图片
#                     jpg_files.extend([(record_id, file) for file in jpg_file])
#                 elif start_id == item:
#                     # 如果找到start_id文件夹，开始添加图片
#                     flag = True
#                     jpg_files.extend([(record_id, file) for file in jpg_file])
#                 else:
#                     # 如果未找到start_id文件夹，跳过当前文件夹
#                     continue

#                 # 更新剩余需要获取的图片数量
#                 remaining_num -= len(jpg_file)
#                 if remaining_num <= 0:
#                     break

#         if remaining_num <= 0:
#             break

#     # 截取前num个图片，或者所有可用图片
#     frames = jpg_files[:num] if num <= len(jpg_files) else jpg_files

#     return frames



def calculate_distance_between_frames(extrinsic_translation1, extrinsic_translation2):
    # 将 extrinsic_translation 转换为 NumPy 数组
    translation1 = np.array(extrinsic_translation1)
    translation2 = np.array(extrinsic_translation2)
    # 计算两帧之间的欧氏距离
    distance = np.linalg.norm(translation2 - translation1)
    return distance


if __name__ == '__main__':
    start_time = time.time()
    sys.stdout = open('output0.txt', 'w')
    # 加载数据集
    scenes_c = dict()
    scenes_c["special_scene"] = {}
    config_dict = dict( data_dir = "/home/duanqingchuan/lane_3d/to_anno", dataset = "Neolix", pred_pose_dir = "/home/duanqingchuan/lane_3d/to_anno", is_absolute_pose = "True", main_sensor = "top_lidar", pose_mode = "pred_pose", debug = "False", save_dir = "/home/duanqingchuan/lane_3d/result", undistorted_pcd_dir = "", scenes_classification = scenes_c)
    data_loader = DataLoader(config_dict)
    # read_input_dir = "/home/duanqingchuan/lane_3d/to_anno/20230720121852.record.00007/front_left_1"
    read_input_dir = "/home/duanqingchuan/lane_3d/to_anno"
    datasets_data, data_root = data_loader.load_data()
    neo_data = NEOLIX( datasets_data, data_root)
    # 整理数据集打印 #
    # pp = pprint.PrettyPrinter(indent=4, width=80)
    # print('!!!!')
    # pp.pprint(datasets_data)

    # 输入序列
    sensor_list = ['front_left_1']
    all_frames = input_seq( 200, '1689827215.600000_1689827215.545000', read_input_dir, sensor_list)
    # print('all_frames: ')
    # print(all_frames)
    frames = []
    for seq_id, frame_id in all_frames:
        temp_frame = frame(seq_id, frame_id)
        frames.append(temp_frame)

    # 初始化序列中每帧图像和lidar基本信息
    neo_data.load_frame(frames)

    # 逐帧标注
    for one_frame in frames:
        # 将2D标注画到图中
        neo_data.project_lanes_to_image(one_frame)
        cv2.imwrite('/home/duanqingchuan/lane_3d/result_100/img_anno_lanes_{}.jpg'.format(one_frame.frame_id), cv2.cvtColor(one_frame.img_buf, cv2.COLOR_BGR2RGB))
        # 3D标注
        neo_data.project_lidar_to_image(frames, one_frame)
        cv2.imwrite('/home/duanqingchuan/lane_3d/result_100/img_label_lanes_{}.jpg'.format(one_frame.frame_id), cv2.cvtColor(one_frame.img_buf, cv2.COLOR_BGR2RGB))
    
    end_time = time.time()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    elapsed_time = end_time - start_time
    print(f"代码运行时间：{elapsed_time:.4f} 秒")