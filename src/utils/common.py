#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   common.py
@Time    :   2021/12/05 12:35:57
@Author  :   chenminghua 
'''

def compute_iou(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x,y,w,h)      (x,y)代表矩形左上的顶点，（w,h）代表矩形的宽和高。下同。
    :param rec2: (x,y,w,h)
    :return: 交并比IOU.
    """
    rec1 = [rec1[0], rec1[1], rec1[0]+rec1[2], rec1[1]+rec1[3]]
    rec2 = [rec2[0], rec2[1], rec2[0]+rec2[2], rec2[1]+rec2[3]]
    # print(rec1, rec2)
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
