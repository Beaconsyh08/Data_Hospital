
import json
import cv2
import argparse
import os
import subprocess
import numpy as np
from tqdm import tqdm
import math
from PIL import Image, ImageDraw, ImageFont


class TopViewer(object):
    GRID_COLOR = (72, 72, 72, 255)
    GRID_THICKNESS = 2

    def __init__(self, x_range=60, y_range=30, view_unit=0.1) -> None:
        super().__init__()
        """
        x_range: 纵向最远距离
        y_range: 横向最远距离
        view_unit: 每一个pixel代表的距离
        """
        self.x_range = x_range
        self.y_range = y_range
        self.view_unit = view_unit
        self.initTopView()

    def world2Image(self, cord):
        return int(cord / self.view_unit)

    def initTopView(self):
        self.view_h = 2 * self.world2Image(self.x_range)
        self.view_w = 2 * self.world2Image(self.y_range)
        self.view_ctx = (self.view_w // 2, self.view_h // 2)
        self.view_mat = np.zeros((self.view_h, self.view_w, 3), dtype=np.uint8)
        # create grid
        x_grid = 5 # unit m, 纵向每个方格10米
        y_grid = 5 # unit m, 横向每个方格5米
        x_grid_img_res = self.world2Image(x_grid)
        y_grid_img_res = self.world2Image(y_grid)
        grid_w = self.view_w // y_grid_img_res
        grid_h = self.view_h // x_grid_img_res

        for idx in range(grid_w):
            point_s = (idx * y_grid_img_res, 0)
            point_e = (idx * y_grid_img_res, self.view_h)
            cv2.circle(self.view_mat, point_s, 1, self.GRID_COLOR, thickness=self.GRID_THICKNESS)
            cv2.circle(self.view_mat, point_e, 1, self.GRID_COLOR, thickness=self.GRID_THICKNESS)
            cv2.line(self.view_mat, point_s, point_e, self.GRID_COLOR, thickness=self.GRID_THICKNESS)
        
        for idy in range(grid_h):
            point_s = (0, idy * x_grid_img_res)
            point_e = (self.view_w, idy * x_grid_img_res)
            cv2.circle(self.view_mat, point_s, 1, self.GRID_COLOR, thickness=self.GRID_THICKNESS)
            cv2.circle(self.view_mat, point_e, 1, self.GRID_COLOR, thickness=self.GRID_THICKNESS)
            cv2.line(self.view_mat, point_s, point_e, self.GRID_COLOR, thickness=self.GRID_THICKNESS)            

        self.drawSelfCar()
        return self.view_mat

    def drawSelfCar(self):
        cv2.circle(self.view_mat, self.view_ctx, 5, (0, 0, 255), thickness=2)

    def world2view(self, point):
        x, y = point
        return (int(self.view_ctx[0] + x / self.view_unit), int(self.view_ctx[1] + y / self.view_unit))

    def get_rotate_rectangle_vertices_points(self, center_x, center_y, length, width, yaw):
        b = math.cos(yaw) * 0.5
        a = math.sin(yaw) * 0.5
        
        pt0 = [center_x - a*width - b*length, center_y + b*width - a*length]
        pt1 = [center_x + a*width - b*length, center_y - b*width - a*length]
        pt2 = [2*center_x - pt0[0], 2*center_y - pt0[1]]
        pt3 = [2*center_x - pt1[0], 2*center_y - pt1[1]]
        ptyaw = [center_x + b*length, center_y + a*length]

        pts = [pt0, pt1, pt2, pt3]
        return pts,ptyaw
        
    def drawCameraObject(self, camera_position, vehicle2camera: tuple):
        v2c_R, v2c_T = vehicle2camera
        v2c_R = np.asmatrix(v2c_R)
        v2c_T = np.asmatrix(v2c_T)
        camera_position = np.array(camera_position).reshape(3, 1)
        vehicle_position = np.matmul(v2c_R.I, camera_position) - np.matmul(v2c_R.I, v2c_T)
        vehicle_x = vehicle_position[0, 0]
        vehicle_y = vehicle_position[1, 0]
        vehicle_z = vehicle_position[2, 0]

        bev_position = (-vehicle_y, -vehicle_x)
        viewer_position = self.world2view(bev_position)
        cv2.circle(self.view_mat, viewer_position, 5, (0, 255, 255), thickness=10)
        return self.view_mat
    
    def drawVehicleObject(self, vehicle_position, object_dimension, vehicle_yaw, color=(0, 0, 255), thickness=2,orient=True):
        """
        vehicle_position: 障碍物在车辆坐标系下的中心点坐标
        object_dimension: 障碍物的长宽高
        vehicle_yaw: 障碍物车辆坐标系下的yaw角弧度
        """
        vehicle_x, vehicle_y, _ = vehicle_position

        # 画旋转框
        _, length, width = object_dimension
        # pts = self.get_rotate_rectangle_vertices_points(vehicle_x, vehicle_y, length, width, vehicle_yaw - math.pi/2)
        pts,ptyaw = self.get_rotate_rectangle_vertices_points(vehicle_x, vehicle_y, length, width, vehicle_yaw)
        
        bev_position_0 = (-pts[0][1], -pts[0][0])
        viewer_position_0 = self.world2view(bev_position_0)
        bev_position_1 = (-pts[1][1], -pts[1][0])
        viewer_position_1 = self.world2view(bev_position_1)
        bev_position_2 = (-pts[2][1], -pts[2][0])
        viewer_position_2 = self.world2view(bev_position_2)
        bev_position_3 = (-pts[3][1], -pts[3][0])
        viewer_position_3 = self.world2view(bev_position_3)

        bev_center = (-vehicle_y,-vehicle_x)
        viewer_position_center = self.world2view(bev_center)
        bev_position_yaw = (-ptyaw[1],-ptyaw[0])
        viewer_position_yaw = self.world2view(bev_position_yaw)
        
        cv2.line(self.view_mat, viewer_position_0, viewer_position_1, color, thickness=thickness, lineType=4)
        cv2.line(self.view_mat, viewer_position_1, viewer_position_2, color, thickness=thickness, lineType=4)
        cv2.line(self.view_mat, viewer_position_2, viewer_position_3, color, thickness=thickness, lineType=4)
        cv2.line(self.view_mat, viewer_position_3, viewer_position_0, color, thickness=thickness, lineType=4)
        if orient:
            cv2.line(self.view_mat, viewer_position_center, viewer_position_yaw, color, thickness=thickness, lineType=4)

        return self.view_mat
    
    def drawLidarObject(self, lidar_position, object_dimension, lidar_yaw, lidar2vehicle: tuple,lv_yaw):
        l2v_R, l2v_T = lidar2vehicle
        l2v_R = np.asmatrix(l2v_R)
        l2v_T = np.asmatrix(l2v_T)
        lidar_position = np.array(lidar_position).reshape(3, 1)
        vehicle_position = np.matmul(l2v_R, lidar_position) + l2v_T
        vehicle_x = vehicle_position[0, 0]
        vehicle_y = vehicle_position[1, 0]
        vehicle_z = vehicle_position[2, 0]

        vehicle_yaw = lidar_yaw - lv_yaw

        self.drawVehicleObject((vehicle_x,vehicle_y,vehicle_z), object_dimension, vehicle_yaw, color=(0,255,0))
        
        # bev_position = (-vehicle_y, -vehicle_x)
        # viewer_position = self.world2view(bev_position)
        # cv2.circle(self.view_mat, viewer_position, 5, (0, 255, 255), thickness=10)
        return self.view_mat