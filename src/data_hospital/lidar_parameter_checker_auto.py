from tqdm import tqdm
import json
import os
from itertools import groupby
import open3d as o3d
import cv2
import numpy as np
from pyquaternion import Quaternion
import random
from concurrent.futures import ProcessPoolExecutor
import math
from pypcd import pypcd


class ColorScalesManager(object):
    def __init__(self) -> None:
        self.scale = 1.0
        self.display_range = [0.0, 1.0]
        self.saturation_range = [0.0, 1.0]

        self.color_scale_mode = 'blue->green->yellow->red'
        self.color_table = []
        self.color_table_scale_map = {
            'blue->green->yellow->red':[['0000FF', 0.0], ['00FF00',0.33], ['FFFF00', 0.66], ['FF0000',1.0]],
            'dip direct(repeat)[0-360]':[]
        }
        pass

    def init_color_table(self, color_scale_mode='blue->green->yellow->red'):
        self.color_scale_mode = color_scale_mode
        if color_scale_mode not in self.color_table_scale_map:
            print(['[error] color_scale_mode[', color_scale_mode, '] is error'])
            exit(0)
        color_table = self.__create_color_table(self.color_table_scale_map[color_scale_mode])
        self.color_table = color_table

        # self.draw_color_table(color_table)

    def __create_color_table(self, color_scales=[]):
        ds = 1.0/255
        color_table = []
        for idx in range(len(color_scales)-1):
            v1 = int(color_scales[idx][0], 16)
            [r1, g1, b1] = self.__color_to_tuple(v1)
            s1 = color_scales[idx][1]
            v2 = int(color_scales[idx+1][0], 16)
            [r2, g2, b2] = self.__color_to_tuple(v2)
            s2 = color_scales[idx+1][1]

            s = s1
            while (s < s2):
                r = (s - s1)/(s2 - s1) * (r2 - r1) + r1
                g = (s - s1)/(s2 - s1) * (g2 - g1) + g1
                b = (s - s1)/(s2 - s1) * (b2 - b1) + b1
                s += ds
                color_table.append([int(b), int(g), int(r)])
        return color_table[:256]
    
    def __color_to_tuple(self, c):
        r = c//256//256
        g = (c-r*256*256)// 256
        b = c - r * 256 * 256 - g * 256
        return[r, g, b]

    def draw_color_table(self, color_table):
        color_bar = np.zeros((20, 257, 3),dtype=np.uint8)

        for idx, c in enumerate(color_table):
            color_bar[:, idx, :] = c
        cv2.imshow('color bar', color_bar)
        cv2.waitKey(0)

    def set_display_range(self, min_v, max_v):
        '''
        min_v max_v = [0, 1]
        '''
        self.display_range = [min_v, max_v]
        pass

    def set_saturation_rage(self, min_v, max_v):
        '''
        min_v max_v = [0, 1]
        '''
        self.saturation_range = [min_v, max_v]
        pass
    
    def check_intensity(self, intensity):
        if np.max(intensity) > 1:
            self.scale = 1/255.0

    def get_color(self, v):
        '''
        v = [0, 1]
        '''
        v = min(max(v*self.scale, self.display_range[0]), self.display_range[1])
        [s1, s2] = self.saturation_range
        v = min(max(v, s1), s2)

        r = (v - s1)/(s2 - s1) * (1.0 - 0.0) + 0.0
        r = int(r * 255)
        # print(v, r, len(self.color_table))
        return self.color_table[r]
    
    
def transform_matrix(
        translation: np.ndarray = np.array([0, 0, 0]),
        rotation: Quaternion = Quaternion([1, 0, 0, 0]),
        inverse: bool = False,
) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def parse_sensor_param(sensor_param):
    quat = sensor_param["pose"]["attitude"]
    qx = quat.get("x", 0)
    qy = quat.get("y", 0)
    qz = quat.get("z", 0)
    qw = quat.get("w", 0)
    trans = sensor_param["pose"]["translation"]
    if trans is None:
        tx = ty = tz = 0
    else:
        tx = trans.get("x", 0)
        ty = trans.get("y", 0)
        tz = trans.get("z", 0)
    sensor2ego_pose = transform_matrix(
        np.array([tx, ty, tz]), Quaternion([qw, qx, qy, qz])
    )
    return sensor2ego_pose


class CameraParameters:
    def __init__(
            self,
    ) -> None:
        self.width_ = None
        self.height_ = None
        self.intrinsic_ = None
        self.distortion_ = None
        self.pose_ = None

    def read_camera_params(self, camera_params, camera_name):
        camera_name = camera_name.replace("_record", "")
        camera_param = None
        if isinstance(camera_params, list):
            for param in camera_params:
                if "name" in param.keys() and param["name"] == camera_name:
                    camera_param = param
                    break
        elif isinstance(camera_params, dict):
            camera_param = camera_params.get(camera_name, None)
        else:
            raise ValueError("dont support ")

        if camera_param is None:
            print("total camera names: ", [cam["name"] for cam in camera_params])
            raise ValueError("not found camera_name %s" % camera_name)
        fx = camera_param["fx"]
        fy = camera_param["fy"]
        cx = camera_param["cx"]
        cy = camera_param["cy"]
        distortion = camera_param["distortion"]
        width = camera_param.get("image_width", 1920)
        height = camera_param.get("image_height", 1080)
        pose = parse_sensor_param(camera_param)

        intrinsic = np.eye(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy

        distortion = np.array(distortion, np.float32)
        self.distortion_ = distortion
        self.height_ = height
        self.width_ = width
        self.intrinsic_ = intrinsic
        self.pose_ = pose
        
        
def mat2vec(mat):
    rotation = mat[:3, :3]
    # rvec, _ = cv2.Rodrigues(rotation)
    tvec = mat[:3, -1].reshape(3, 1)
    return rotation, tvec


def prase_sensor_config(calibration_file, lidar_name="MIDDLE_LIDAR"):
    cam_params = {}
    sensor_config = calibration_file["sensor_config"]
    for cam_conf in sensor_config["cam_param"]:
        single_cam_param = CameraParameters()
        if cam_conf.get("name", None) is None:
            continue
        if cam_conf.get("pose", None) is None:
            continue
        single_cam_param.read_camera_params(
            sensor_config["cam_param"], cam_conf["name"])
        cam_params[cam_conf["name"]] = single_cam_param
    exist_flag = False
    for param in sensor_config["lidar_param"]:
        if "name" in param.keys() and param["name"] == lidar_name:
            lidar_param_dict = param
            exist_flag = True
            break
    assert exist_flag, "lidar_front is not exist in hardware config"
    lidar_params = parse_sensor_param(lidar_param_dict)
    sensors_params = {"camera": cam_params, "lidar": lidar_params}
    return sensors_params


def draw_cloud(image, project_cloud, intensity=None, ind=None):
    [h, w] = image.shape[:2]
    # points_2d, zs = project_cloud
    project_cloud = project_cloud.astype(np.int32)
    
    # if ind is None:
    #     ind = (zs >= 0.5) & (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) \
    #         & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    
    # points_2d = points_2d[ind, :]
    if intensity is not None:
        # intensity_tmp = intensity[ind]
        # print(intensity)
        color_scales_manager = ColorScalesManager()
        color_scales_manager.init_color_table()
        color_scales_manager.set_display_range(0.0, 1.0)
        color_scales_manager.set_saturation_rage(0.0, 0.21336615)
        # color_scales_manager.set_saturation_rage(0.0, 0.6)
        color_scales_manager.check_intensity(intensity)
        for idx, point in enumerate(project_cloud):
            color = color_scales_manager.get_color(intensity[idx])
            # print(intensity_tmp[idx], color)
            cv2.circle(image, tuple(point[0]), 1, tuple(color))
    else:
        for idx, point in enumerate(point[0]):
            # print(intensity_tmp[idx], color)
            cv2.circle(image, tuple(point[0]), 1, (0, 0, 255))
    return image


# def calculate_ratio(lane_intensity_matrix, lane_seg_matrix):


def main(pcd_file, hw_conf, image_file, camera_orientation, output_path):
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    calibration_file = json.loads(open(hw_conf).read())
    sensors_params = prase_sensor_config(calibration_file)
    pcd_points = np.array(o3d.io.read_point_cloud(pcd_file).points)
    # print(pcd_points.shape)
    # pcd_points = np.array([[22., 0., -2.]])
    # print(pcd_points.shape)
    camera_parameters = sensors_params["camera"][camera_orientation]
    extrinsic = np.linalg.inv(camera_parameters.pose_) @ sensors_params["lidar"]
    intrinsic = camera_parameters.intrinsic_
    # dist = camera_parameters.distortion_
    shape = (camera_parameters.height_, camera_parameters.width_)
    dist = camera_parameters.distortion_.tolist()
    if len(dist) == 4:
        dist = dist
    elif len(dist) == 5:
        dist = dist[0:2] + dist[3:] + [dist[2]]
    elif len(dist) == 8:
        dist = dist[0:2] + dist[3:5] + [dist[2]] + dist[5:]
    else:
        dist = dist
    rvec, tvec = mat2vec(extrinsic)
    # ret = cv2.projectPoints(pcd_points, rvec, tvec, intrinsic, np.array(dist))
    # from IPython import embed;embed()
    lidar_points_camera = np.dot(rvec, pcd_points.T).T + tvec.reshape(1, 3)
    indices = np.where(lidar_points_camera[:, 2] > 0)[0]
    lidar_points_camera = lidar_points_camera[lidar_points_camera[:, 2] > 0]
    ret = cv2.projectPoints(lidar_points_camera, np.eye(3), np.zeros((3, 1)), intrinsic, np.array(dist))
    img = cv2.imread(image_file)
    canvas = np.zeros_like(img)
    h, w = shape[:2]
    intersity = pypcd.PointCloud.from_path(pcd_file).pc_data['intensity'][indices]
    img = draw_cloud(image=img, project_cloud=ret[0], intensity=intersity)
    # for i, idx in enumerate(ret[0]):
    #     x, y = idx[0]
    #     if 0 < x < w and 0 < y < h:
    #         dis = math.dist(pcd_points[indices[i]][:2], [0, 0])
    #         if dis < 20:
    #             canvas[int(y), int(x), :] = (255, 0, 0)
    #         elif dis < 50:
    #             canvas[int(y), int(x), :] = (0, 0, 255)
    #         elif dis < 100:
    #             canvas[int(y), int(x), :] = (0, 255, 0)
    #         else:
    #             canvas[int(y), int(x), :] = (40, 120, 200)
                
    # for x in range(int(w)):
    #     for y in range(int(h)):
    #         if canvas[y][x].tolist() != [0, 0, 0]:
    #             color = tuple(canvas[y][x].tolist())
    #             cv2.circle(img, (x, y), 2, color, 1)
    # cv2.imwrite(os.path.join(output_path, os.path.basename(image_file)), img)
    cv2.imwrite(output_path, img)


def process_data(data):
    pcd_file = data["point_clouds"][0]
    hw_conf = data["hardware_config"]
    image_file = data["images"][0]
    folder_name = data["carday_id"]
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(image_file))
    # try:
    main(pcd_file, hw_conf, image_file, camera_orientation, output_path)
    # except:

if __name__ == '__main__':
    input_path = "/root/data-juicer/outputs/demo-gn6/demo-processed.jsonl"
    camera_orientation = "front_middle_camera"
    output_dir = "/mnt/share_disk/songyuhao/data/lidar_projection_all_new"
    n = 10
    datas = []

    with open(input_path, "r") as f:
        for line in f:
            datas.append(json.loads(line))

    # Group data by carday_id
    datas.sort(key=lambda x: x['carday_id'])
    grouped_data = {key: list(group) for key, group in groupby(datas, key=lambda x: x['carday_id'])}
    sdatas = []
    for gdata_key in grouped_data.keys():
        sdatas.append(random.sample(grouped_data[gdata_key], min(n, len(grouped_data[gdata_key]))))

    sdatas = [_ for sublist in sdatas for _ in sublist]
    # for sdata in tqdm(sdatas):
    #     process_data(sdata)

    # Use multiprocessing to parallelize the execution of the main function
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_data, sdatas), total=len(sdatas)))
