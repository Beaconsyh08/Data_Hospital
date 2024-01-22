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
    dist = camera_parameters.distortion_
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
    lidar_points_camera = lidar_points_camera[lidar_points_camera[:, 2] > 0]
    ret = cv2.projectPoints(lidar_points_camera, np.eye(3), np.zeros((3, 1)), intrinsic, np.array(dist))
    img = cv2.imread(image_file)
    canvas = np.zeros_like(img)
    h, w = shape[:2]
    for idx in ret[0]:
        x, y = idx[0]
        if 0 < x < w and 0 < y < h:
            canvas[int(y), int(x), :] = (0, 0, 255)
    for x in range(int(w)):
        for y in range(int(h)):
            if canvas[y][x].tolist() != [0, 0, 0]:
                color = tuple(canvas[y][x].tolist())
                cv2.circle(img, (x, y), 2, color, 1)
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
    try:
        main(pcd_file, hw_conf, image_file, camera_orientation, output_path)
    except:
        print("pcd_file: ", pcd_file)
        print("hw_conf: ", hw_conf)
        print("image_file: ", image_file)
        print("output_folder: ", output_folder)
        print("camera_orientation: ", camera_orientation)

if __name__ == '__main__':
    input_path = "/root/data-juicer/outputs/demo-gn6/demo-processed.jsonl"
    camera_orientation = "front_middle_camera"
    output_dir = "/mnt/share_disk/songyuhao/data/lidar_projection_all"
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

    # Use multiprocessing to parallelize the execution of the main function
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_data, sdatas), total=len(sdatas)))
