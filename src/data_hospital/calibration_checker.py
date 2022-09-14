import shutil
from google.protobuf import text_format
from src.data_hospital.utils.reproject.hardware_config_pb2 import HardwareConfig, CameraParameter, LidarParameter
from src.data_hospital.utils.reproject.reproject_utils import get_mtx, get_RTMatrix, get_minArearect_v2, convert_3dbox_to_8corner, get_iou
from src.data_hospital.utils.reproject.lidar2camera import getCalibTransformMatrix, lidar2camera, vehicle2camera
import json
import numpy as np
import cv2
from tqdm import tqdm
import os
import math
from concurrent.futures import ThreadPoolExecutor
import warnings
from src.data_manager.data_manager import load_from_pickle
warnings.filterwarnings("ignore")
from src.utils.logger import get_logger
import pandas as pd


class CalibrationChecker():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.coor = cfg.COOR
        self.vis = cfg.VIS
        self.vis_path = cfg.VIS_PATH
        
        if self.vis:
            shutil.os.makedirs(self.vis_path, exist_ok=True)
            self.logger.debug("Images Has Been Saved in: %s" % self.vis_path)
            
        self.save_dir = cfg.SAVE_DIR
        self.load_path = cfg.LOAD_PATH
        self.threshold = cfg.THRESHOLD
        self.txt_paths = [self.load_path + "/" + _ for _ in os.listdir(self.load_path)]
        self.res_pd = pd.DataFrame()
        self.prob_lst = []
        

# 获取相机标定参数
    def get_camera_para_json(self, clibraction_data, camera_name):
        for cam in clibraction_data['sensor_config']['cam_param']:
            name = cam.get('name', '')
            if name == camera_name:
                fx = cam['fx']
                fy = cam['fy']
                cx = cam['cx']
                cy = cam['cy']

                # method 3
                try:
                    k1 = cam['distortion'][0]
                    k2 = cam['distortion'][1]
                    k3 = cam['distortion'][2]
                    p1 = cam['distortion'][3]
                    p2 = cam['distortion'][4]
                except:
                    if int(cam['distortion'][3]) == 0:
                        # # method 1
                        k1 = cam['distortion'][0]
                        k2 = cam['distortion'][1]
                        k3 = cam['distortion'][2]
                        p1 = 0
                        p2 = 0
                    else:
                        # method 2
                        k1 = cam['distortion'][0]
                        k2 = cam['distortion'][1]
                        p1 = cam['distortion'][2]
                        p2 = cam['distortion'][3]
                        k3 = 0

                w1 = cam['pose']['attitude'].get('w', 0)
                x1 = cam['pose']['attitude'].get('x', 0)
                y1 = cam['pose']['attitude'].get('y', 0)
                z1 = cam['pose']['attitude'].get('z', 0)
                q = np.array([x1, y1, z1, w1])
                x = cam['pose']['translation'].get('x', 0)
                y = cam['pose']['translation'].get('y', 0)
                z = cam['pose']['translation'].get('z', 0)
                pos = [x, y, z]
                #print('fx = ', fx,fy,cx,cy,k1,k2,k3,w1,x1,y1,z1,x,y,z)
                return fx, fy, cx, cy, k1, k2, k3, p1, p2, q, pos


    def get_camera_para_prototxt(self, clibraction_data, camera_name):
        for cam in clibraction_data.sensor_config.cam_param:
            if camera_name == CameraParameter.CameraNameType.Name(cam.name):
                fx = cam.fx
                fy = cam.fy
                cx = cam.cx
                cy = cam.cy

                # # method 2
                # k1 = cam['distortion'][0]
                # k2 = cam['distortion'][1]
                # p1 = cam['distortion'][2]
                # p2 = cam['distortion'][3]
                # k3 = 0

                # method 3
                try:
                    k1 = cam.distortion[0]
                    k2 = cam.distortion[1]
                    k3 = cam.distortion[2]
                    p1 = cam.distortion[3]
                    p2 = cam.distortion[4]
                except:
                    if cam.distortion[3] == 0:
                        # # method 1
                        k1 = cam.distortion[0]
                        k2 = cam.distortion[1]
                        k3 = cam.distortion[2]
                        p1 = cam.distortion[3]
                        p2 = 0
                    else:
                        # method 2
                        k1 = cam.distortion[0]
                        k2 = cam.distortion[1]
                        p1 = cam.distortion[2]
                        p2 = cam.distortion[3]
                        k3 = 0

                w1 = cam.pose.attitude.w
                x1 = cam.pose.attitude.x
                y1 = cam.pose.attitude.y
                z1 = cam.pose.attitude.z
                q = np.array([x1, y1, z1, w1])
                x = cam.pose.translation.x
                y = cam.pose.translation.y
                z = cam.pose.translation.z
                pos = [x, y, z]
                #print('fx = ', fx,fy,cx,cy,k1,k2,k3,w1,x1,y1,z1,x,y,z)
            # except:
            #     fx,fy,cx,cy,k1,k2,k3,p1,p2,q,pos = 1,1,0,0,0,0,0,0,0,np.zeros(4),[0,0,0]
                return fx, fy, cx, cy, k1, k2, k3, p1, p2, q, pos
        raise ''


    # 根据相机名称获取内外参矩阵和畸变参数
    def get_camera_parameter(self, clibraction_data, camera_name, cali_type='json'):
        if cali_type == 'json':
            fx, fy, cx, cy, k1, k2, k3, p1, p2, q, pos = self.get_camera_para_json(
                clibraction_data, camera_name)
        else:
            fx, fy, cx, cy, k1, k2, k3, p1, p2, q, pos = self.get_camera_para_prototxt(
                clibraction_data, camera_name)
        ext = get_RTMatrix(q, pos)
        # dist=np.array([k1,k2,p1,p2,k3])
        dist = np.array([k1, k2, p1, p2])
        mtx = get_mtx(fx, fy, cx, cy)
        R = ext[:3, :3]
        t = np.array(ext[:3, 3])
        return mtx, R, t, dist


    def find_intesection_point(self, bbox_lines, img_line):
        '''
        bbox_lines: [12,2,2]  12 lines, 2 points, (x,y)
        img_line:   [2,2]     2 points, (x,y)

        def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
            px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
            py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
            return [px, py]
        '''

        # def getLine(start, end):
        #     x1, y1 = start[:,0],start[:,1]
        #     x2, y2 = end[:,0],end[:,1]
        #     return y2 - y1, x1 - x2, x2 * y1 - x1 * y2

        # bbox_a, bbox_b, bbox_c = getLine(bbox_lines[:,0], bbox_lines[:,1])
        # img_a, img_b, img_c = getLine(img_line[0][None,...], img_line[1][None,...])

        # px = (bbox_b * img_c - bbox_c * img_b) / (img_b * bbox_a - img_a * bbox_b)
        # py = (bbox_a * img_c - img_a * bbox_c) / (img_a * bbox_b - bbox_a * img_b)

        x1 = bbox_lines[:, 0, 0]
        y1 = bbox_lines[:, 0, 1]
        x2 = bbox_lines[:, 1, 0]
        y2 = bbox_lines[:, 1, 1]
        x3 = img_line[0][0]
        y3 = img_line[0][1]
        x4 = img_line[1][0]
        y4 = img_line[1][1]

        px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / \
            ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / \
            ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

        inlines = ((px-x3)*(px-x4) <= 0) & ((px-x1)*(px-x2) <=
                                            0) & ((py-y3)*(py-y4) <= 0) & ((py-y1)*(py-y2) <= 0)
        intersect_points = np.concatenate((px[..., None], py[..., None]), axis=1)
        intersect_points = intersect_points[np.where(
            inlines == True)[0]].reshape(-1, 2).astype(int)

        return intersect_points


    def check_in_img(self, points, img_h, img_w, gap=20):
        return (points[:, 0] <= img_w+gap) & (points[:, 1] <= img_h+gap) & (points[:, 0] >= -gap) & (points[:, 1] >= -gap)


    def convert_points_to_croped_image_v2(self, img_points_i, points_in_img_mask, img_h, img_w, gap=20):
        '''
        img_points_i: [8,2]
        out_mask: [8,]
        pairs_points: [24,2,2]
        pairs_mask: [24,2,1]
        '''

        max_point = 100000
        # reset_mask = abs(img_points_i) > max_point
        reset_mask_x = (abs(img_points_i[:, 0]) > max_point) | (
            abs(img_points_i[:, 0]) < (-max_point))
        if reset_mask_x.any():
            u = img_points_i[reset_mask_x][:, 0]
            v = img_points_i[reset_mask_x][:, 1]
            k = v/u
            img_points_i[:, 0][reset_mask_x == True] = max_point*(abs(u)/u)
            img_points_i[:, 1][reset_mask_x == True] = k*max_point*(abs(u)/u)

            reset_mask_y = abs(img_points_i[:, 1]) > max_point
            if reset_mask_y.any():
                u = img_points_i[reset_mask_y][:, 0]
                v = img_points_i[reset_mask_y][:, 1]
                k = v/u
                img_points_i[:, 0][reset_mask_y == True] = max_point*(abs(v)/v) / k
                img_points_i[:, 1][reset_mask_y == True] = max_point*(abs(v)/v)

        img_corners = np.array([   # tl,bl,tr,br
            (-gap, -gap),
            (-gap, img_h+gap),
            (img_w+gap, -gap),
            (img_w+gap, img_h+gap),
        ])

        bbox_lines = np.concatenate(
            (img_points_i[[0, 1]][None, ...],
            img_points_i[[1, 2]][None, ...],
            img_points_i[[2, 3]][None, ...],
            img_points_i[[3, 0]][None, ...],

            img_points_i[[4, 5]][None, ...],
            img_points_i[[5, 6]][None, ...],
            img_points_i[[6, 7]][None, ...],
            img_points_i[[7, 4]][None, ...],

            img_points_i[[0, 4]][None, ...],
            img_points_i[[1, 5]][None, ...],
            img_points_i[[2, 6]][None, ...],
            img_points_i[[3, 7]][None, ...]),

            axis=0)

        img_lines = np.concatenate(
            [  # left,top,right,bottom
                img_corners[[0, 1]][None, ...],
                img_corners[[0, 2]][None, ...],
                img_corners[[2, 3]][None, ...],
                img_corners[[1, 3]][None, ...],
            ], axis=0)

        intersection_points = np.array([]).reshape(0, 2)
        for img_line in img_lines:
            add_points = self.find_intesection_point(bbox_lines, img_line)
            intersection_points = np.concatenate(
                (intersection_points, add_points), axis=0)

        remain_img_points = img_points_i[points_in_img_mask]
        points = np.concatenate((remain_img_points, intersection_points), axis=0)

        if points.shape[0] == 0:
            points = np.array([[0, 0], [0, 0]])

        # (24,), first point of pairs will be removed.
        return points, intersection_points.astype(int), img_points_i

    # find intersection with z=0.1
    def intersection_point(self, p0, p1, z_n):
        p0 = p0.copy()
        p1 = p1.copy()

        return p0[0] - (p0[0] - p1[0]) * (p0[2] - z_n) / (p0[2] - p1[2])


    def draw_rgb_projections(self, image, projections, color=(255, 0, 255), thickness=2, darker=1.0, pad=200):
        # img = (image.copy()*darker).astype(np.uint8)
        img = image
        num = len(projections)
        projections = projections.astype(np.int32)
        # for n in range(num):
        #     # qs = projections[n]
        qs = projections
        for k in range(0, 4):
            i, j = k, (k+1) % 4
            cv2.line(img, (qs[i, 0]+pad, qs[i, 1]+pad), (qs[j, 0] +
                    pad, qs[j, 1]+pad), color, thickness, cv2.LINE_AA)
            # cv2.imwrite('{}_{}.jpg'.format(i,j),img)
            i, j = k+4, (k+1) % 4 + 4
            cv2.line(img, (qs[i, 0]+pad, qs[i, 1]+pad), (qs[j, 0] +
                    pad, qs[j, 1]+pad), color, thickness, cv2.LINE_AA)
            # cv2.imwrite('{}_{}.jpg'.format(i,j),img)
            i, j = k, k+4
            cv2.line(img, (qs[i, 0]+pad, qs[i, 1]+pad), (qs[j, 0] +
                    pad, qs[j, 1]+pad), color, thickness, cv2.LINE_AA)
            # cv2.imwrite('{}_{}.jpg'.format(i,j),img)
        cv2.line(img, (qs[0, 0]+pad, qs[0, 1]+pad), (qs[7, 0] +
                pad, qs[7, 1]+pad), color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3, 0]+pad, qs[3, 1]+pad), (qs[4, 0] +
                pad, qs[4, 1]+pad), color, thickness, cv2.LINE_AA)

        # for i,pt in enumerate(projections):
        # cv2.circle(img,pt,10,(255,0,0),-1)
        # cv2.putText(img, str(i), (pt[0]+20,pt[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        return img

    # 根据相机名称把对应标注文件中的3dbbox投影到2d图像，计算iou值
    def get_objects_iou(self, each_json_path_m, tag, img_count=0, set_orientation=''):
        with open(each_json_path_m, 'r', encoding='utf-8') as load_f:
            strF = load_f.read()
        if len(strF) > 0:
            data_ori = json.loads(strF)

        camera_name = data_ori.get('camera_orientation')
        if set_orientation != '' and camera_name != set_orientation:
            for view in data_ori.get('relative_images_data', data_ori.get('relative_sensors_data')):
                if view.get('camera_orientation', '') == set_orientation:
                    # if view['camera_orientation'] == 'front_middle_camera':
                    new_json_path = '/'+view['image_json']
                    with open(new_json_path, 'r') as f:
                        data_ori = json.load(f)
                        camera_name = set_orientation
                    break
        # if set_orientation != '' and camera_name != set_orientation:
        #     continue

        objects = data_ori.get('objects', [])
        try:
            height = int(data_ori.get('height'))
            width = int(data_ori.get('width'))
        except:
            img = cv2.imread('/'+data_ori['imgUrl'])
            height = img.shape[0]
            width = img.shape[1]

        gap = 1
        z_n = 0.5
        cls_list = ['car', 'bus', 'truck', 'van']

        clib_path = '/'+data_ori['calibration_file']
        if '.json' in clib_path:
            with open(clib_path, 'r', encoding='utf-8') as f_input:
                clibraction_data = json.load(f_input)
            mtx, R, t, dist = self.get_camera_parameter(
                clibraction_data, camera_name, cali_type='json')
        else:
            with open(clib_path, 'rb') as fcalib:
                clibraction_data = HardwareConfig()
                text_format.Merge(fcalib.read(), clibraction_data)
            mtx, R, t, dist = self.get_camera_parameter(
                clibraction_data, camera_name, cali_type='prototxt')

        if self.vis:
            img = cv2.imread('/'+data_ori['imgUrl'])
            # img = cv2.undistort(img,mtx,dist)
            # cv2.imwrite('undistorted img.jpg',img)
            img_3dbbox = (img.copy()).astype(np.uint8)
            pad = 200
            img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)),
                        'constant', constant_values=(255, 255))  # h-y, w-x, c
            img_3dbbox = np.pad(img_3dbbox, ((pad, pad), (pad, pad), (0, 0)),
                                'constant', constant_values=(255, 255))  # h-y, w-x, c

        if tag == 'label':
            lv_param, vc_param, cv_param, cam_intric = getCalibTransformMatrix(
                clib_path, camera_name, lidar_name='FRONT_LIDAR', cali_file=True)
            # lidar_R,lidar_t=get_lidar_parameter(clibraction_data)

        croped_img_w = width
        croped_img_h = height

        ious = []
        ious_cal = []
        for obj_info in objects:

            # obj=obj_info.get('lidar_coor')
            if tag in ['inference', 'label']:
                cls = obj_info['className']
                box2d_t = obj_info['bbox'].copy()
                w, h = obj_info['bbox'][2:]
                cx = obj_info['bbox'][0] + w/2
                if tag == 'inference':
                    obj = obj_info.get('camera_coor')
                else:
                    obj = obj_info.get('3D_attributes')
            elif tag == 'cam_tensor':
                cls = 'car'
                xmin, ymin, xmax, ymax = obj_info[:4]
                w, h = max((xmax - xmin), 0), max((ymax - ymin), 0)
                cx = (xmax - xmin)/2
                box2d_t = [xmin, ymin, w, h]
                h_dim = obj_info[5]
                l_dim = obj_info[6]
                w_dim = obj_info[7]
                x = obj_info[8]
                y = obj_info[9]
                z = obj_info[10]
                heading = obj_info[11]
                obj = [0, 0]
            else:
                raise ""

            if cls not in cls_list:
                ious.append(1.0)
                continue

            if w * h < 20*20:
                ious.append(1.0)
                continue
            if (abs(cx) < croped_img_w/16) or (abs(cx-croped_img_w) < croped_img_w/16):
                ious.append(1.0)
                continue
            box2d_t[2] = box2d_t[2]+box2d_t[0]
            box2d_t[3] = box2d_t[3]+box2d_t[1]
            tlx, tly, brx, bry = int(box2d_t[0]), int(
                box2d_t[1]), int(box2d_t[2]), int(box2d_t[3])

            if obj != None and len(obj) > 0:
                if tag == 'inference':  # in vehicle coordinate
                    x = obj['position'].get('x')
                    y = obj['position'].get('y')
                    z = obj['position'].get('z')
                    heading = obj['rotation']['yaw']
                    l = obj['dimension']['length']
                    w = obj['dimension']['width']
                    h = obj['dimension']['height']
                elif tag == 'label':  # in lidar coordinate
                    # transfer to vehicle coor. methed 1:
                    l = obj['dimension']['length']
                    w = obj['dimension']['width']
                    h = obj['dimension']['height']
                    lidar_position = [
                        obj['position']['x'],
                        obj['position']['y'],
                        obj['position']['z']]
                    lidar_yaw = obj['rotation']['yaw']
                    lidar_yaw_point = [lidar_position[0] + math.cos(lidar_yaw) * l, 
                    lidar_position[1] + math.sin(lidar_yaw) * l, 
                    lidar_position[2]]

                    if self.coor == "Lidar":
                        camera_position = lidar2camera(
                            lidar_position, lv_param, vc_param)
                        camera_yaw_point = lidar2camera(
                            lidar_yaw_point, lv_param, vc_param)
                    elif self.coor == "Vehicle":
                        camera_position = vehicle2camera(
                            lidar_position, vc_param)
                        camera_yaw_point = vehicle2camera(
                            lidar_yaw_point, vc_param)
                    else:
                        self.logger.error("Please Choose Coor between Car and Lidar.")
                        
                    x = camera_position[0]
                    y = camera_position[1]
                    z = camera_position[2]
                    heading = math.atan2(
                        camera_yaw_point[2]-camera_position[2], camera_yaw_point[0]-camera_position[0])
                elif tag == 'cam_tensor':
                    w = w_dim
                    h = h_dim
                    l = l_dim

                if abs(x) > 30 or abs(y) > 30:
                    ious.append(1.0)
                    continue
                arr = np.array([x, y, z, heading, l, w, h])
                worldpoint1 = convert_3dbox_to_8corner(arr, coor='camera')
                # print('arr:',arr)

                cam_z_tag = worldpoint1[:, 2] <= 0
                if np.sum(cam_z_tag) == 0:
                    pass
                elif np.sum(cam_z_tag) >= 5:
                    worldpoint1 = np.ones([8, 3])
                else:
                    if np.sum(cam_z_tag[:4]) > 0:
                        if np.sum(cam_z_tag[:4]) == 1:
                            if np.where(cam_z_tag[:4] == True)[0] in [0, 1]:
                                x_n = self.intersection_point(
                                    worldpoint1[0], worldpoint1[1], z_n)
                                if worldpoint1[0][2] <= 0:
                                    worldpoint1[0] = np.array(
                                        [x_n, worldpoint1[1][1], z_n])
                                else:
                                    worldpoint1[1] = np.array(
                                        [x_n, worldpoint1[0][1], z_n])
                            else:
                                x_n = self.intersection_point(
                                    worldpoint1[2], worldpoint1[3], z_n)
                                if worldpoint1[2][2] <= 0:
                                    worldpoint1[2] = np.array(
                                        [x_n, worldpoint1[3][1], z_n])
                                else:
                                    worldpoint1[3] = np.array(
                                        [x_n, worldpoint1[2][1], z_n])

                        elif np.sum(cam_z_tag[:4]) == 2:
                            if (np.where(cam_z_tag[:4] == True)[0] == [0, 1]).all():
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0], worldpoint1[3], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[1], worldpoint1[2], z_n)
                                worldpoint1[0] = np.array(
                                    [x_n_0, worldpoint1[3][1], z_n])
                                worldpoint1[1] = np.array(
                                    [x_n_1, worldpoint1[2][1], z_n])
                            elif (np.where(cam_z_tag[:4] == True)[0] == [1, 2]).all():
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0], worldpoint1[1], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[3], worldpoint1[2], z_n)
                                worldpoint1[1] = np.array(
                                    [x_n_0, worldpoint1[0][1], z_n])
                                worldpoint1[2] = np.array(
                                    [x_n_1, worldpoint1[3][1], z_n])
                            elif (np.where(cam_z_tag[:4] == True)[0] == [2, 3]).all():
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0], worldpoint1[3], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[1], worldpoint1[2], z_n)
                                worldpoint1[3] = np.array(
                                    [x_n_0, worldpoint1[0][1], z_n])
                                worldpoint1[2] = np.array(
                                    [x_n_1, worldpoint1[1][1], z_n])
                            elif (np.where(cam_z_tag[:4] == True)[0] == [0, 3]).all():
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0], worldpoint1[1], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[3], worldpoint1[2], z_n)
                                worldpoint1[0] = np.array(
                                    [x_n_0, worldpoint1[1][1], z_n])
                                worldpoint1[3] = np.array(
                                    [x_n_1, worldpoint1[2][1], z_n])

                        else:
                            raise ""
                    if np.sum(cam_z_tag[4:]) > 0:
                        if np.sum(cam_z_tag[4:]) == 1:
                            if np.where(cam_z_tag[4:] == True)[0] in [0, 1]:
                                x_n = self.intersection_point(
                                    worldpoint1[0+4], worldpoint1[1+4], z_n)
                                if worldpoint1[0+4][2] <= 0:
                                    worldpoint1[0 +
                                                4] = np.array([x_n, worldpoint1[1+4][1], z_n])
                                else:
                                    worldpoint1[1 +
                                                4] = np.array([x_n, worldpoint1[0+4][1], z_n])
                            else:
                                x_n = self.intersection_point(
                                    worldpoint1[2+4], worldpoint1[3+4], z_n)
                                if worldpoint1[2+4][2] <= 0:
                                    worldpoint1[2 +
                                                4] = np.array([x_n, worldpoint1[3+4][1], z_n])
                                else:
                                    worldpoint1[3 +
                                                4] = np.array([x_n, worldpoint1[2+4][1], z_n])

                        elif np.sum(cam_z_tag[4:]) == 2:
                            if np.where(cam_z_tag[4:] == True)[0].tolist() == [0, 1]:
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0+4], worldpoint1[3+4], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[1+4], worldpoint1[2+4], z_n)
                                worldpoint1[0 +
                                            4] = np.array([x_n_0, worldpoint1[3+4][1], z_n])
                                worldpoint1[1 +
                                            4] = np.array([x_n_1, worldpoint1[2+4][1], z_n])
                            elif np.where(cam_z_tag[4:] == True)[0].tolist() == [1, 2]:
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0+4], worldpoint1[1+4], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[3+4], worldpoint1[2+4], z_n)
                                worldpoint1[1 +
                                            4] = np.array([x_n_0, worldpoint1[0+4][1], z_n])
                                worldpoint1[2 +
                                            4] = np.array([x_n_1, worldpoint1[3+4][1], z_n])
                            elif np.where(cam_z_tag[4:] == True)[0].tolist() == [2, 3]:
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0+4], worldpoint1[3+4], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[1+4], worldpoint1[2+4], z_n)
                                worldpoint1[3 +
                                            4] = np.array([x_n_0, worldpoint1[0+4][1], z_n])
                                worldpoint1[2 +
                                            4] = np.array([x_n_1, worldpoint1[1+4][1], z_n])
                            elif np.where(cam_z_tag[4:] == True)[0].tolist() == [0, 3]:
                                x_n_0 = self.intersection_point(
                                    worldpoint1[0+4], worldpoint1[1+4], z_n)
                                x_n_1 = self.intersection_point(
                                    worldpoint1[3+4], worldpoint1[2+4], z_n)
                                worldpoint1[0 +
                                            4] = np.array([x_n_0, worldpoint1[1+4][1], z_n])
                                worldpoint1[3 +
                                            4] = np.array([x_n_1, worldpoint1[2+4][1], z_n])

                        else:
                            raise ""

                # dist = np.zeros((5,))
                t = np.zeros(3)
                R = np.eye(3)
                imagePoints, jacobian = cv2.projectPoints(
                    worldpoint1, R, t, mtx, dist)
                imagePoints = np.reshape(imagePoints, (8, 2)).astype(int)

                points_in_img_mask = self.check_in_img(
                    imagePoints, croped_img_h, croped_img_w, gap=gap)
                if np.sum(~points_in_img_mask) > 0:
                    bbox2d, added_points, imagePoints = self.convert_points_to_croped_image_v2(
                        imagePoints, points_in_img_mask, croped_img_h, croped_img_w, gap=gap)
                    if self.vis:
                        for pt in added_points:
                            cv2.circle(img_3dbbox, pt+pad, 10, (255, 0, 0), -1)
                else:
                    bbox2d = imagePoints

                bbox_pj = get_minArearect_v2(bbox2d)

                iou = get_iou(box2d_t, bbox_pj)
                ious.append(iou)
                ious_cal.append(iou)
                if self.vis:
                    # cv2.circle(img_3dbbox,imagePoints[0][0].astype(int),20,(255,0,0),-1)
                    # cv2.imwrite('./inf_3dbbox.jpg',img_3dbbox)
                    img_3dbbox = self.draw_rgb_projections(
                        img_3dbbox, imagePoints, pad=pad)
                    cv2.rectangle(img, (int(bbox_pj[0]+pad), int(bbox_pj[1]+pad)), (int(
                        bbox_pj[2]+pad), int(bbox_pj[3]+pad)), (0, 0, 255), 2)
                    cv2.rectangle(img, (tlx+pad, tly+pad),
                                (brx+pad, bry+pad), (0, 255, 0), 2)
                    cv2.putText(img, f'{iou:.2f}', (tlx+pad, tly+pad-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        if self.vis:
            # img = cv2.undistort(img,mtx,dist,None,mtx)
            if img_count == 'basename':
                cardid = each_json_path_m.split('/')[-3]
                basename = each_json_path_m.split('/')[-1].split('.')[0]
                img_count = cardid + '_' + basename
            cv2.imwrite("%s/%s.jpg" % (self.vis_path, img_count), img)
            cv2.imwrite("%s/%s_3dbbox.jpg" % (self.vis_path, img_count), img_3dbbox)
        return ious, ious_cal


    def multi_process(self, file_path, tag, img_count):
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            count = len(lines)
        with ThreadPoolExecutor(max_workers=64) as pool:
            temp_infors = [pool.submit(self.get_objects_iou, json_path, tag,
                                    img_count=img_count) for json_path in lines]
            temp_infors = temp_infors
            data_infos = [t.result()[1] for t in temp_infors]

        ious_cal = [sum(ious_cal_frame)/len(ious_cal_frame)
                    for ious_cal_frame in data_infos if len(ious_cal_frame) > 0]
        frame_low_ious = []
        for i, frame in enumerate(data_infos):
            if len(frame) != 0:
                mean_iou = sum(frame)/len(frame)
                if mean_iou < 0.2:
                    frame_low_ious.append(lines[i]+'\n')
        # with open('%s/low_iou.txt' % save_dir, 'w') as f:
        #     f.writelines(frame_low_ious)
        mean_ious = sum(ious_cal)/len(ious_cal) if len(ious_cal) > 0 else 0
        return mean_ious, count

    
    def output_result(self,):
        clean_path = "%s/clean.txt" % self.save_dir
        prob_path = "%s/prob.txt" % self.save_dir
        with open(clean_path, "w") as clean_file:
            with open(prob_path, "w") as prob_file:
                clean, prob = 0, 0
                for i, row in self.res_pd.iterrows():
                    key, value = row["Txt"], row["Mean Iou"]
                    if value > self.threshold:
                        with open(key) as input_file:
                            for line in input_file:
                                clean_file.write(line)
                                clean += 1
                    else:
                        with open(key) as input_file:
                            for line in input_file:
                                prob_file.write(line)
                                self.prob_lst.append(line)
                                prob += 1
        
        self.logger.debug("\nClean Frame: %d \nProb Frame: %d" % (clean, prob))
        self.logger.debug("Cleaned Path Has Been Saved in: %s" % clean_path)
        excel_path = "%s/result.xlsx" % self.save_dir
        self.res_pd.to_excel(excel_path)
        hist_path = "%s/result_hist.png" % self.save_dir
        self.res_pd["Mean Iou"].hist(bins=20).get_figure().savefig(hist_path)
        self.logger.debug("Result Stats Has Been Saved in: %s" % excel_path)
    
    
    def save_dataframes(self,):
        self.df = load_from_pickle(self.cfg.DATAFRAME_PATH)
        self.df["calibration_error"] = 0
        
        df_error = self.df[self.df.json_path.isin(self.prob_lst)]
        
        df_error["calibration_error"] = 1
        cali_dict = df_error.calibration_error.to_dict()
        self.df['calibration_error'] = [cali_dict.get(_, None) for _ in self.df.index]
        
        self.save_to_pickle(self.df, self.cfg.DATAFRAME_PATH)
        
    
    def diagnose(self,):
        # tag = 'label'  # 'inference'
        tag = 'label'
        img_count = 'basename'  # 0
        txt_lst, iou_lst, count_lst = [], [], []
        
        for txt in tqdm(self.txt_paths):
            mean_iou, count = self.multi_process(txt, tag, img_count)
            txt_lst.append(txt)
            iou_lst.append(mean_iou)
            count_lst.append(count)
        
        self.res_pd["Txt"] = txt_lst
        self.res_pd["Mean Iou"] = iou_lst
        self.res_pd["Count"] = count_lst
        
        self.output_result()

        
