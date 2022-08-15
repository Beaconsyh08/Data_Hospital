import argparse
import json
import os
import math
from pathlib import Path
import shutil
from tqdm import tqdm
from tools.coor_trans.ann_parser import transfer_attr
from tools.coor_trans.ann_parser import getCalibTransformMatrix, lidar2camera, lidar2vehicle, vehicle2lidar
from multiprocessing import Process,Queue,Pool,Pipe,Manager
from src.utils.logger import get_logger
import glob



# @DATASETS.register_module()
class CoorTransDoctor():
    CAMERA_MAPS = dict(
        FRONT_MIDDLE_CAMERA = ['front_wide_camera', 'front_middle_camera'],
        FRONT_LEFT_CAMERA = ['lf_wide_camera', 'front_left_camera'],
        FRONT_RIGHT_CAMERA = ['rf_wide_camera', 'front_right_camera'],
        REAR_MIIDLE_CAMERA = ['rear_wide_camera', 'rear_middle_camera'],
        REAR_LEFT_CAMERA = [],
        REAR_RIGHT_CAMERA = []
    )

    CLASSES = ('car','bus','truck','pedestrian','rider', 'bicycle','tricycle')

    # 车辆的方向
    VEHICLE_DIRECTIONS = ['left', 'right', 'front', 'behind', 'left_front', 'left_behind', 'right_front', 'right_behind']
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.input_path = cfg.INPUT_PATH
        self.output_path = cfg.OUTPUT_PATH
        self.logger = get_logger()
        
        self.lv_trans = None
        self.vc_trans = None
        # lidar坐标系到车辆坐标系rotation中的yaw
        # self.lv_yaw = None
        self.Tcv_params = None
        self.CLASSES = ('car', 'bus', 'truck', 'pedestrian', 'rider', 'bicycle', 'tricycle')
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        
    def parsetUnion3D(self, obj_ann,test=False):
        if not test:
            union3d_dict = dict(
                union3d_flag=0,
                height=0,
                length=0,
                width=0,
                ct_x=0,
                ct_y=0,
                ct_z=0,
                pitch=0,
                roll=0,
                yaw=0
            )
        else:
            union3d_dict = dict(
            union3d_flag=0,
            height=0,
            length=0,
            width=0,
            ct_x=0,
            ct_y=0,
            ct_z=0,
            pitch=0,
            roll=0,
            yaw=0,
            occlusion=9,
            crowding=9,)
            attrs_2d = obj_ann.get("2D_attributes", None)
            if attrs_2d is not None and len(attrs_2d) == 0:
                attrs_2d = None
            if attrs_2d is not None:
                union3d_dict['occlusion'] = transfer_attr(attrs_2d.get('occlusion', None))
                union3d_dict['crowding'] = transfer_attr(attrs_2d.get('crowding', None))

        attrs_3d = obj_ann.get("3D_attributes", None)
        if attrs_3d is not None and len(attrs_3d) == 0:
            attrs_3d = None
        if attrs_3d is not None:
            union3d_dict['union3d_flag'] = 1
            try:
                union3d_dict['height'] = attrs_3d['dimension']['height']
                union3d_dict['length'] = attrs_3d['dimension']['length']
                union3d_dict['width'] = attrs_3d['dimension']['width']

                lidar_position = [
                    attrs_3d['position']['x'],
                    attrs_3d['position']['y'],
                    attrs_3d['position']['z']]

                lidar_yaw = attrs_3d['rotation']['yaw']
                lidar_yaw_point = [lidar_position[0]+math.cos(lidar_yaw)*union3d_dict['length'],lidar_position[1]+math.sin(lidar_yaw)*union3d_dict['length'],lidar_position[2]]

                #超过120米的物体不预测lidar 3D信息, lidar坐标系V71C001和VV7C001前期数据x朝右，y朝前，z朝上
                if math.sqrt(lidar_position[0]**2 + lidar_position[1]**2) > 100:
                    union3d_dict['union3d_flag'] = 0


                    ## qa_自带vehicle坐标系
                    # union3d_dict['ct_x'] = lidar_position[0]
                    # union3d_dict['ct_y'] = lidar_position[1]
                    # union3d_dict['ct_z'] = lidar_position[2]
                if test:
                    # 获取车辆坐标系下的坐标, 车辆坐标系x朝前，y朝左，z朝上
                    vehicle_position = lidar2vehicle(lidar_position, self.lv_trans)
                    vehicle_yaw_point = lidar2vehicle(lidar_yaw_point, self.lv_trans)
                    union3d_dict['ct_x'] = vehicle_position[0]
                    union3d_dict['ct_y'] = vehicle_position[1]
                    union3d_dict['ct_z'] = vehicle_position[2]
                    union3d_dict['pitch'] = attrs_3d['rotation']['pitch']
                    union3d_dict['roll'] = attrs_3d['rotation']['roll']
                    union3d_dict['yaw'] = math.atan2(vehicle_yaw_point[1]-vehicle_position[1],vehicle_yaw_point[0]-vehicle_position[0])
                    
                    ## qa_自带vehicle坐标系
                    # union3d_dict['ct_x'] = lidar_position[0]
                    # union3d_dict['ct_y'] = lidar_position[1]
                    # union3d_dict['ct_z'] = lidar_position[2]
                    # union3d_dict['pitch'] = attrs_3d['rotation']['pitch']
                    # union3d_dict['roll'] = attrs_3d['rotation']['roll']
                    # union3d_dict['yaw'] = attrs_3d['rotation']['yaw']

                else:
                    # 获取相机坐标系下的坐标, 相机坐标系x朝右，y朝下，z朝前
                    camera_position = lidar2camera(lidar_position, self.lv_trans, self.vc_trans)
                    camera_yaw_point = lidar2camera(lidar_yaw_point, self.lv_trans, self.vc_trans)

                    union3d_dict['ct_x'] = camera_position[0]
                    union3d_dict['ct_y'] = camera_position[1]
                    union3d_dict['ct_z'] = camera_position[2]
                    union3d_dict['pitch'] = attrs_3d['rotation']['pitch']
                    union3d_dict['roll'] = attrs_3d['rotation']['roll']
                    union3d_dict['yaw'] = math.atan2(camera_yaw_point[2]-camera_position[2], camera_yaw_point[0]-camera_position[0])
            except Exception:
                print(attrs_3d)
        return union3d_dict

    def get_ann_info(self, json_ann_path:str):
        with open(json_ann_path, 'r') as f:
            json_map =  json.load(f)

        image_width = json_map['width']
        image_height = json_map['height']

        camera_orientation = json_map.get('camera_orientation', 'front_wide_camera')
        if json_map.get('calibration_file', None):
            self.lv_trans, self.vc_trans, self.Tcv_params = getCalibTransformMatrix(
                json_ann_path, camera_orientation)

        for idx, obj_ann in enumerate(json_map['objects']):
            if not obj_ann:
                continue
            # TODO: 去除非车辆的类别
            # if obj_ann['className'].lower() not in self.CLASSES:
            #     continue
            x, y, w, h = obj_ann['bbox']

            try:
                if w < 0 or h < 0 or w > (image_width+20) or h > (image_height+20) or x < 0 or y < 0 or x > (image_width+20) or y > (image_width+20):
                    continue
                if (x + w) > (image_width+20) or (y + h) > (image_height+20):
                    continue
                if (x + 0.5 * w) >= image_width or (y + 0.5 * h) >= image_height:
                    continue
            except Exception:
                continue

            inter_w = max(0, min(x + w, image_width))
            inter_h = max(0, min(y + h, image_height))
            if 0 == inter_w * inter_h:
                continue
            # 小于15*15的障碍物不需要
            # if w * h <= 15*15 or w < 1 or h < 1:
            #     continue
            bbox = [x, y, x + w, y + h]
            ann = self.parsetUnion3D(obj_ann, test=True)
            try:
                json_map['objects'][idx]['3D_attributes']['position']['x'] = ann['ct_x']
                json_map['objects'][idx]['3D_attributes']['position']['y'] = ann['ct_y']
                json_map['objects'][idx]['3D_attributes']['position']['z'] = ann['ct_z']
                json_map['objects'][idx]['3D_attributes']['rotation']['yaw'] = ann['yaw']
            except Exception:
                # print('fail to save vehicle_coor:', ann)
                continue
        # json_save_path = os.path.join('/cpfs/output/gt', Path(json_ann_path).name)
        
        # json_save_path = json_ann_path
        # self.write_json(json_save_path, json_map)
        return json_map
            
    def write_json(self, json_file, json_map):
        """
        dump json file
        """
        shutil.os.makedirs(Path(json_file).parent, exist_ok=True)
        with open(json_file, 'w') as fp:
            fp.write(json.dumps(json_map ,indent=2))


    def coor_trans(self, num, lines):
        for idx, line in enumerate(lines):
            json_path = line.strip()
            json_map = self.get_ann_info(json_path)
            json_map["ori_path"] = json_path
            # print(json_map)
            save_new_json_path = os.path.join(self.output_dir, str(idx)+'_'+num+'.json')
            self.write_json(save_new_json_path, json_map)

    def splitPaths_MultiProcessing(self, json_paths, coreNum):
        lenPerSt= int(len(json_paths)/coreNum+1)
        json_inf_each = []
        for i in range(coreNum):
            json_inf_each.append(json_paths[i*lenPerSt:(i+1)*lenPerSt])
        # recieve the return values of multi-processing
        jobs = []
        for i in range(coreNum):
            p = Process(target=self.coor_trans, args=(str(i), json_inf_each[i]))
            jobs.append(p)
            p.start()
        for proc in tqdm(jobs, desc="Coordinate transforming"):
            proc.join()
            
            
    def generate_txt(self,):
        with open (self.output_path, "w") as output_file:
            for root, directories, file in os.walk(self.output_dir):
                for file in file:
                    save_path = os.path.join(root, file)
                    output_file.writelines(save_path + "\n")
            
            self.logger.debug('Saving Final Txt: %s' % self.output_path)
            
            
    def diagnose(self,):
        with open(self.input_path,  'r') as f:
            lines = f.readlines()
        
        coreNum = 40
        self.splitPaths_MultiProcessing(lines, coreNum)
        self.logger.debug('Saving Trans Result: %s' % (self.output_dir))
        self.generate_txt()
        
