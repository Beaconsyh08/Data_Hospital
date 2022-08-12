# This is the test scripts for union3d annotation
import json
import os
import numpy as np
from google.protobuf import text_format
from scipy.spatial.transform import Rotation as R
from src.data_hospital.utils.reproject.hardware_config_pb2 import HardwareConfig, CameraParameter, LidarParameter

current_path = os.path.split(os.path.realpath(__file__))[0]
new_hw_config_path = os.path.join(current_path, 'hw_pb/hw_config_VV7C001_20211023.prototxt')

CARD_HW_ERROR_MAPS = (
    dict(
        hw_config=new_hw_config_path,
        card_ids=[
            "6145b13840935aa81c765080",
            "6145b13840935aa81c765083",
            "6145b13840935aa81c765089",
            "6145b13840935aa81c76508c",
            "6145b1fe40935aa81c7754b7",
            "6145b1fe40935aa81c7754ba",
            "6145b1fe40935aa81c7754c0",
            "6145b1fe40935aa81c7754c3",
            "6145b2b440935aa81c7848d3",
            "6145b2b440935aa81c7848d6",
            "6145b2b440935aa81c7848dc",
            "6145b2b440935aa81c7848df",
            "614ac63040935aa81c077a60",
            "6150563f316f405f23516e2a",
            "61505706c29c8a4cbcd3ba19",
            "615057d8c29c8a4cbcd3d3bd",
            "6150582bc29c8a4cbcd3d3c6",
            "6151aa8dc29c8a4cbca1c6d2",
            "6151ad2dd7da30e50a9b6999",
            "61541a2ec29c8a4cbc522eea",
            "61541bb8d7da30e50a5d2b81",
            "61541ca3c29c8a4cbc58136a",
            "61541db8c29c8a4cbc5a7a4b",
            "61544954c29c8a4cbc8a1869",
            "61544c28d7da30e50a989634",
            "61545227d7da30e50a989656",
            "6154571cc29c8a4cbc8a5f1f",
            "615457f9c29c8a4cbc90bc1b",
            "61545927c29c8a4cbc93cf60",
            "61545ae9d7da30e50a995b2b",
            "61552067d7da30e50ab53991",
            "61552156c29c8a4cbcbe5c3d",
            "615521f2c29c8a4cbcbf30a4",
            "615526bfc29c8a4cbccaf38f",
            "615528a2d7da30e50ac75c4f",
            "615ff38bb45d5fc1d17af9c2",
            "616559d4ef8bc40fb9224e07",
            "6166873bef8bc40fb903cce1"
        ]
    ),
)

def getCalibTransformMatrix(json_file, camera_name, lidar_name="FRONT_LIDAR",cali_file=False, cali_dict=False):
    if not cali_file and not cali_dict:
        with open(json_file, 'r') as fjson:
            json_cnt = json.load(fjson)
            calib_file = f"/{json_cnt['calibration_file']}"
            # 有些卡片的标定文件是错误的，更新标定文件
            # for hw_error in CARD_HW_ERROR_MAPS:
            #     for card_id in hw_error['card_ids']:
            #         # if card_id in json_file:
            #         if card_id in json_cnt:
            #             calib_file = hw_error['hw_config']
            #             break
    elif cali_file:
        calib_file = json_file
    elif cali_dict:
        calib_file = ''
    else:
        raise TypeError
    
    Tvc_params = None
    Tlv_params = None
    Tcv_params = None
    if calib_file.endswith('.json'):
        with open(calib_file, 'r') as fjson:
            json_cnt = json.load(fjson)

            sensor_cfg = json_cnt['sensor_config']
            for cam_param in sensor_cfg['cam_param']:
                if camera_name == cam_param.get('name', None):
                    cam_intric = [[cam_param['fx'],0,cam_param['cx']],[0,cam_param['fy'],cam_param['cy']],[0,0,1]]
                    cam_pose = cam_param['pose']
                    cam_attitude = cam_pose['attitude']
                    cam_quat = [cam_attitude.get('x', 0), cam_attitude.get('y', 0),
                                cam_attitude.get('z', 0), cam_attitude.get('w', 0)]
                    cv_R = R.from_quat(cam_quat)
                    vc_R = cv_R.inv().as_matrix()
                    cam_translation = cam_pose['translation']
                    cv_T = np.array([cam_translation.get('x', 0), cam_translation.get('y', 0),
                                     cam_translation.get('z', 0)]).reshape(3, 1)
                    vc_T = -1 * np.matmul(vc_R, cv_T)
                    Tvc_params = (vc_R, vc_T)
                    Tcv_params = (cv_R.as_matrix(),cv_T)
                    Tcv_yaw = cam_pose['attitude_ypr']['yaw']
                    break
            
            # 有些车辆128线雷达，叫做FRONT_LIDAR, 有些叫做MIDDLE_LIDAR
            for lidar_param in sensor_cfg['lidar_param']:
                if lidar_param.get('name', None) == 'MIDDLE_LIDAR':
                    lidar_name = 'MIDDLE_LIDAR'
                    break
            for lidar_param in sensor_cfg['lidar_param']:
                if lidar_name == lidar_param.get('name', None):
                    lidar_pose = lidar_param['pose']
                    lidar_attitude = lidar_pose['attitude']
                    lidar_quat = [lidar_attitude.get('x', 0), lidar_attitude.get('y', 0),
                                  lidar_attitude.get('z', 0), lidar_attitude.get('w', 0)]
                    lv_R = R.from_quat(lidar_quat).as_matrix()
                    lidar_translation = lidar_pose['translation']
                    lv_T = np.array([lidar_translation.get('x', 0), lidar_translation.get('y', 0),
                                    lidar_translation.get('z', 0)]).reshape(3, 1)
                    Tlv_params = (lv_R, lv_T)
                    # lidar 2 vehicle yaw
                    lv_yaw = lidar_pose['attitude_ypr']['yaw']
                    break
    elif calib_file.endswith('.prototxt'):
        with open(calib_file, 'rb') as fcalib:
            hw_cfg = HardwareConfig()
            text_format.Merge(fcalib.read(), hw_cfg)
            for cam_param in hw_cfg.sensor_config.cam_param:
                if camera_name == CameraParameter.CameraNameType.Name(cam_param.name):
                    cam_intric = [[cam_param.fx,0,cam_param.cx],[0,cam_param.fy,cam_param.cy],[0,0,1]]
                    cam_pose = cam_param.pose
                    cam_attitude = cam_pose.attitude
                    cam_quat = [cam_attitude.x, cam_attitude.y,
                                cam_attitude.z, cam_attitude.w]
                    cv_R = R.from_quat(cam_quat)
                    vc_R = cv_R.inv().as_matrix()
                    cv_T = np.array([cam_pose.translation.x, cam_pose.translation.y,
                                     cam_pose.translation.z]).reshape(3, 1)
                    vc_T = -1 * np.matmul(vc_R, cv_T)
                    Tvc_params = (vc_R, vc_T)
                    Tcv_params = (cv_R.as_matrix(),cv_T)
                    break
            for lidar_param in hw_cfg.sensor_config.lidar_param:
                if lidar_name == LidarParameter.LidarName.Name(lidar_param.name):
                    lidar_pose = lidar_param.pose
                    lidar_attitude = lidar_pose.attitude
                    lidar_quat = [lidar_attitude.x, lidar_attitude.y,
                                  lidar_attitude.z, lidar_attitude.w]
                    lv_R = R.from_quat(lidar_quat).as_matrix()
                    lv_T = np.array([lidar_pose.translation.x,
                                     lidar_pose.translation.y,
                                     lidar_pose.translation.z]).reshape(3, 1)
                    Tlv_params = (lv_R, lv_T)
                    lv_yaw = lidar_pose.attitude_ypr.yaw
                    break
    elif cali_dict:
        json_cnt = json_file
        sensor_cfg = json_cnt['sensor_config']
        for cam_param in sensor_cfg['cam_param']:
            if camera_name == cam_param.get('name', None):
                cam_intric = [[cam_param['fx'],0,cam_param['cx']],[0,cam_param['fy'],cam_param['cy']],[0,0,1]]
                cam_pose = cam_param['pose']
                cam_attitude = cam_pose['attitude']
                cam_quat = [cam_attitude.get('x', 0), cam_attitude.get('y', 0),
                            cam_attitude.get('z', 0), cam_attitude.get('w', 0)]
                cv_R = R.from_quat(cam_quat)
                vc_R = cv_R.inv().as_matrix()
                cam_translation = cam_pose['translation']
                cv_T = np.array([cam_translation.get('x', 0), cam_translation.get('y', 0),
                                    cam_translation.get('z', 0)]).reshape(3, 1)
                vc_T = -1 * np.matmul(vc_R, cv_T)
                Tvc_params = (vc_R, vc_T)
                Tcv_params = (cv_R.as_matrix(),cv_T)
                Tcv_yaw = cam_pose['attitude_ypr']['yaw']
                break
        
        # 有些车辆128线雷达，叫做FRONT_LIDAR, 有些叫做MIDDLE_LIDAR
        for lidar_param in sensor_cfg['lidar_param']:
            if lidar_param.get('name', None) == 'MIDDLE_LIDAR':
                lidar_name = 'MIDDLE_LIDAR'
                break
        for lidar_param in sensor_cfg['lidar_param']:
            if lidar_name == lidar_param.get('name', None):
                lidar_pose = lidar_param['pose']
                lidar_attitude = lidar_pose['attitude']
                lidar_quat = [lidar_attitude.get('x', 0), lidar_attitude.get('y', 0),
                                lidar_attitude.get('z', 0), lidar_attitude.get('w', 0)]
                lv_R = R.from_quat(lidar_quat).as_matrix()
                lidar_translation = lidar_pose['translation']
                lv_T = np.array([lidar_translation.get('x', 0), lidar_translation.get('y', 0),
                                lidar_translation.get('z', 0)]).reshape(3, 1)
                Tlv_params = (lv_R, lv_T)
                # lidar 2 vehicle yaw
                lv_yaw = lidar_pose['attitude_ypr']['yaw']
                break
    else:
        raise RuntimeError("Invalid calibration file ext, ", calib_file)

    if None == Tvc_params or None == Tlv_params or Tcv_params == None:
        raise RuntimeError("Invalid calibration params")
    return Tlv_params, Tvc_params, Tcv_params, cam_intric

def getCalibTransformMatrixFromCalibFile(calib_file, camera_name, lidar_name="FRONT_LIDAR"):
    Tvc_params = None
    Tlv_params = None
    if calib_file.endswith('.json'):
        with open(calib_file, 'r') as fjson:
            json_cnt = json.load(fjson)
            sensor_cfg = json_cnt['sensor_config']
            for cam_param in sensor_cfg['cam_param']:
                if "name" in cam_param and camera_name == cam_param['name']:
                    cam_pose = cam_param['pose']
                    cam_attitude = cam_pose['attitude']
                    cam_quat = [cam_attitude['x'], cam_attitude['y'],
                                cam_attitude['z'], cam_attitude['w']]
                    cv_R = R.from_quat(cam_quat)
                    vc_R = cv_R.inv().as_matrix()
                    cam_translation = cam_pose['translation']
                    cv_T = np.array([cam_translation['x'], cam_translation['y'],
                                     cam_translation['z']]).reshape(3, 1)
                    vc_T = -1 * np.matmul(vc_R, cv_T)
                    Tvc_params = (vc_R, vc_T)
                    break
            for lidar_param in sensor_cfg['lidar_param']:
                if lidar_name == lidar_param['name']:
                    lidar_pose = lidar_param['pose']
                    lidar_attitude = lidar_pose['attitude']
                    lidar_quat = [lidar_attitude.get('x', 0), lidar_attitude.get('y', 0),
                                  lidar_attitude['z'], lidar_attitude['w']]
                    lv_R = R.from_quat(lidar_quat).as_matrix()
                    lidar_translation = lidar_pose['translation']
                    lv_T = np.array([lidar_translation.get('x', 0),
                                     lidar_translation.get('y', 0),
                                     lidar_translation['z']]).reshape(3, 1)
                    Tlv_params = (lv_R, lv_T)
                    break
    elif calib_file.endswith('.prototxt'):
        with open(calib_file, 'rb') as fcalib:
            hw_cfg = HardwareConfig()
            text_format.Merge(fcalib.read(), hw_cfg)
            for cam_param in hw_cfg.sensor_config.cam_param:
                if camera_name == CameraParameter.CameraNameType.Name(cam_param.name):
                    cam_pose = cam_param.pose
                    cam_attitude = cam_pose.attitude
                    cam_quat = [cam_attitude.x, cam_attitude.y,
                                cam_attitude.z, cam_attitude.w]
                    cv_R = R.from_quat(cam_quat)
                    vc_R = cv_R.inv().as_matrix()
                    cv_T = np.array([cam_pose.translation.x, cam_pose.translation.y,
                                     cam_pose.translation.z]).reshape(3, 1)
                    vc_T = -1 * np.matmul(vc_R, cv_T)
                    Tvc_params = (vc_R, vc_T)
                    break
            for lidar_param in hw_cfg.sensor_config.lidar_param:
                if lidar_name == LidarParameter.LidarName.Name(lidar_param.name):
                    lidar_pose = lidar_param.pose
                    lidar_attitude = lidar_pose.attitude
                    lidar_quat = [lidar_attitude.x, lidar_attitude.y,
                                  lidar_attitude.z, lidar_attitude.w]
                    lv_R = R.from_quat(lidar_quat).as_matrix()
                    lv_T = np.array([lidar_pose.translation.x,
                                     lidar_pose.translation.y,
                                     lidar_pose.translation.z]).reshape(3, 1)
                    Tlv_params = (lv_R, lv_T)
                    break
    else:
        raise RuntimeError("Invalid calibration file ext, ", calib_file)

    if None == Tvc_params or None == Tlv_params:
        raise RuntimeError("Invalid calibration params")
    return (lv_R, lv_T), (vc_R, vc_T)

def lidar2camera(position: list, lv_trans: tuple, vc_trans: tuple):
    # print("*******************")
    lidar_position = np.array(position).reshape(3, 1)
    # print("Lidar: ", lidar_position)
    # First, lidar2vechicle
    lv_R, lv_T = lv_trans
    vehicle_position = np.matmul(lv_R, lidar_position) + lv_T
    # print("Vehicle: ", vehicle_position)
    # Second, vehicle2camera
    vc_R, vc_T = vc_trans
    camera_postion = np.matmul(vc_R, vehicle_position) + vc_T
    # print("Camera: ", camera_postion)
    return list(camera_postion.reshape(-1))

def lidar2vehicle(position: list, lv_trans: tuple):
    lidar_position = np.array(position).reshape(3, 1)
    # lidar2vehicle
    lv_R, lv_T = lv_trans
    vehicle_position = np.matmul(lv_R, lidar_position) + lv_T

    return list(vehicle_position.reshape(-1))

def vehicle2camera(vehicle_position: list, Tvc_params: tuple):
    vehicle_position = np.array(vehicle_position).reshape(3, 1)
    vc_R, vc_T = Tvc_params
    camera_postion = np.matmul(vc_R, vehicle_position) + vc_T
    # print("Camera: ", camera_postion)
    return list(camera_postion.reshape(-1))

def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R