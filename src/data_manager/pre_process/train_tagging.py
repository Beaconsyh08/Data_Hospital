import datetime
import math
import pandas as  pd
from configs.config import Config
from src.utils.logger import get_logger
import src.data_manager.data_manager as dm
from src.stats.stats_common import check_truncation
from src.utils.struct import parse_obs
TYPE_MAP = {'car': 'car', 'van': 'car', 
            'truck': 'truck', 'forklift': 'truck',
            'bus':'bus', 
            'rider':'rider',
            'rider-bicycle': 'rider', 'rider-motorcycle':'rider', 
            'rider_bicycle': 'rider', 'rider_motorcycle':'rider',
            'bicycle': 'bicycle', 'motorcycle': 'bicycle',
            'tricycle': 'tricycle', 'closed-tricycle':'tricycle', 'open-tricycle': 'tricycle', 
            'closed_tricycle':'tricycle', 'open_tricycle': 'tricycle', 'pedestrian': 'pedestrian',
           }
def process(df) -> None:
    # load datasets
    df = df[df.priority == 'P0']
    df = df[~df.index.duplicated(keep='first')]

    # obstacle type
    # df['class_name'] = df['class_name'].map(TYPE_MAP)
    df['tag_type'] = df['class_name'].map(TYPE_MAP)
    df['class_name'] = df['class_name'].map(TYPE_MAP)
    df = df[~df['class_name'].isna()]

    # obstacle velocity
    # TODO

    # obstacle heading
    df.loc[(df.yaw > math.pi), 'yaw'] = df.loc[(df.yaw > math.pi)].yaw - math.pi
    df['tag_heading'] = None
    df['yaw'] *= 180 / math.pi
    yaw = df.yaw
    df.loc[((yaw < -45) & (yaw > -135)), 'tag_heading'] = 'forward'
    df.loc[((yaw > 45 ) & (yaw < 135)), 'tag_heading'] = 'reverse'
    df.loc[(((yaw < 45)& (yaw > -45)) | ((yaw > 135) | (yaw < -135))), 'tag_heading'] = 'transverse'

    # obstacle distance
    df['pow'] =  df['x']**2 + df['y']**2
    df["self_dis"] = df["pow"].apply(np.sqrt) #
    df['tag_distance'] = None
    df.loc[df['self_dis'].between(0, 5), 'tag_distance'] = '0-5'
    df.loc[df['self_dis'].between(5, 10),'tag_distance' ] = '5-10'
    df.loc[df['self_dis'].between(10, 20), 'tag_distance'] = '10-20'
    df.loc[df['self_dis'].between(20, 60), 'tag_distance'] = '20-60'
    df.loc[df['self_dis'].between(60, 200), 'tag_distance'] = '60-200'

    # truncation
    for row in tqdm(df.itertuples(), total=len(df)):
        obs = parse_obs(row)
        state = check_truncation_train(obs)
        if state:
            df.at[row.Index, 'tag_truncation'] = state
        else:
            df.at[row.Index, 'tag_truncation'] = 'no_truncation'
    
    # obstacle position
    df['tag_area'] =  None
    df.loc[(df.x > 1) & (df.y < -2.5), 'tag_area'] = 'front_left'
    df.loc[(abs(df.x) < 1) & (df.y < -2.5), 'tag_area'] = 'front'
    df.loc[(df.x < -1) & (df.y < -2.5), 'tag_area'] = 'front_right'
    df.loc[(df.x > 0) & (abs(df.y) < 2.5), 'tag_area'] = 'left'
    df.loc[(df.x < 0) & (abs(df.y) < 2.5), 'tag_area'] = 'right'
    df.loc[(df.x > 1) & (df.y > 2.5), 'tag_area'] = 'rear_left'
    df.loc[(abs(df.x) < 1) & (df.y > 2.5), 'tag_area'] = 'rear'
    df.loc[(df.x < -1) & (df.y > 2.5), 'tag_area'] = 'rear_right'
    return df

    # time
    # df['tag_time'] = df['time'].apply(lambda x: 'night' if x >=datetime.time(19,00,00) or x <=datetime.time(7,00,00) else 'daytime')