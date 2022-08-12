import numpy as np
import math
from numpy import copy

#根据四元素获取旋转矩阵
def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=q.dtype)
    #rot_matrix=np.linalg.inv(rot_matrix)
    return rot_matrix

#获取旋转和平移矩阵
def get_RTMatrix(q,pos):
    RT=np.zeros((4,4))
    rot_matrix=quaternion_to_rotation_matrix(q)
    RT[:3,:3]=rot_matrix
    #t=np.array([-pos[0],-pos[1],-pos[2]])
    #ret=np.dot(rot_matrix,t)
    RT[0,3]=pos[0]
    RT[1,3]=pos[1]
    RT[2,3]=pos[2]
    RT[3,3]=1
    RT=np.linalg.inv(RT)
    return RT
#内参矩阵
def get_mtx(fx,fy,cx,cy):
    mtx=np.zeros((3,3))
    mtx[0,0]=fx
    mtx[0,2]=cx
    mtx[1,1]=fy
    mtx[1,2]=cy
    mtx[2,2]=1
    return mtx

#绕y轴旋转的坐标变换，输入为yaw角
def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,   0],
                     [0,   0,   1]])

#计算iou
def get_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def get_minArearect_v2(box2d):
    box2d=np.int32(box2d)
    # rect = cv2.minAreaRect(box2d)
    # box = cv2.boxPoints(rect)
    # bbox=[np.min(box[:,0]),np.min(box[:,1]),np.max(box[:,0]),np.max(box[:,1])]

    x_min = np.min(box2d[:,0])
    x_max = np.max(box2d[:,0])
    y_min = np.min(box2d[:,1])
    y_max = np.max(box2d[:,1])
    return [x_min,y_min,x_max,y_max]


#根据[x,y,z,l,h,w,yaw]得到3dbbox的8个顶点坐标
def convert_3dbox_to_8corner(bbox3d_input, coor):
    # compute rotational matrix around yaw axis
    bbox3d = copy(bbox3d_input)
    R = roty(bbox3d[3]) 
    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    # # 3d bounding box corners
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    # y_corners = [w/2,w/2,w/2,w/2,-w/2,-w/2,-w/2,-w/2]
    # z_corners = [h/2,-h/2,-h/2,h/2,h/2,-h/2,-h/2,h/2]
    # # rotate and translate 3d bounding box
    # corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # #print corners_3d.shape
    # corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    # corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    # corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]


    r = math.sqrt((l*l)/4+(w*w)/4)
    x_c,y_c,z_c = bbox3d[:3]
    angle_0 = bbox3d[3]           - math.atan(w/l)
    angle_1 = bbox3d[3] + math.pi + math.atan(w/l)
    angle_2 = bbox3d[3] + math.pi - math.atan(w/l)
    angle_3 = bbox3d[3]           + math.atan(w/l)

    if coor == 'camera':
        p0 = [r * math.cos(angle_0) + x_c, y_c + h/2,  r * math.sin(angle_0) + z_c]
        p1 = [r * math.cos(angle_1) + x_c, y_c + h/2,  r * math.sin(angle_1) + z_c]
        p2 = [r * math.cos(angle_2) + x_c, y_c + h/2,  r * math.sin(angle_2) + z_c]
        p3 = [r * math.cos(angle_3) + x_c, y_c + h/2,  r * math.sin(angle_3) + z_c]

        p4 = [r * math.cos(angle_0) + x_c, y_c - h/2,  r * math.sin(angle_0) + z_c]
        p5 = [r * math.cos(angle_1) + x_c, y_c - h/2,  r * math.sin(angle_1) + z_c]
        p6 = [r * math.cos(angle_2) + x_c, y_c - h/2,  r * math.sin(angle_2) + z_c]
        p7 = [r * math.cos(angle_3) + x_c, y_c - h/2,  r * math.sin(angle_3) + z_c]
    elif coor == 'vehicle' or coor == 'lidar':
        p0 = [r * math.cos(angle_0) + x_c, r * math.sin(angle_0) + y_c,   z_c - h/2]
        p1 = [r * math.cos(angle_1) + x_c, r * math.sin(angle_1) + y_c,   z_c - h/2]
        p2 = [r * math.cos(angle_2) + x_c, r * math.sin(angle_2) + y_c,   z_c - h/2]
        p3 = [r * math.cos(angle_3) + x_c, r * math.sin(angle_3) + y_c,   z_c - h/2]

        p4 = [r * math.cos(angle_0) + x_c, r * math.sin(angle_0) + y_c,   z_c + h/2]
        p5 = [r * math.cos(angle_1) + x_c, r * math.sin(angle_1) + y_c,   z_c + h/2]
        p6 = [r * math.cos(angle_2) + x_c, r * math.sin(angle_2) + y_c,   z_c + h/2]
        p7 = [r * math.cos(angle_3) + x_c, r * math.sin(angle_3) + y_c,   z_c + h/2]

    corners_3d = np.array([
        [p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0]],
        [p0[1],p1[1],p2[1],p3[1],p4[1],p5[1],p6[1],p7[1]],
        [p0[2],p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2]]
    ])
    return np.transpose(corners_3d)