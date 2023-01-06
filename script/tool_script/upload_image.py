import sys
sys.path.append("/share/GANs/tools/haomo_ai_framework")
from haomoai.cards import cardapi
import glob
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json

def create_card(target_dir,media_name,project_name="icu30"):
    card_id = ""
    op = cardapi.CardOperation()
    card_id = op.new_card_id(project_name, media_name)
    print("card id:", card_id)
    names = os.listdir(target_dir)
    names = [name for name in names if os.path.isfile(os.path.join(target_dir, name))]
    file_paths = [os.path.join(target_dir, name) for name in names]
    print("find files:", len(names))
    oss_paths = op.upload_files(file_paths=file_paths, names=names, oss_folder=card_id)
    print(op.append_card(card_id, project_name, media_name, oss_paths))
    return card_id, target_dir

def parse_args():
    parser = argparse.ArgumentParser(description='haomo detection inference')
    parser.add_argument('--txt_path', help='the path of translated content data txt', default="")
    parser.add_argument('--infer_path', help='the path of GAN inf', default="/share/2d-od/lei/snowdata/hokkaido_snow")
    parser.add_argument('--media_name', type=str,help='new card media name', default="snow")
    parser.add_argument('--project_name', default="icu30", type=str,help='new card project name')
    parser.add_argument('--fake_name', type=str,help='faken image name', default="")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    target_dir = args.infer_path
    print("===========image card:===================")
    card_id, target_dir = create_card(target_dir,args.media_name)
    card_engine = cardapi.CardOperation()
    new_img_osses = card_engine.get_oss_paths(card_id,args.project_name,args.media_name)
    save_path = "/share/2d-od/lei/snowdata/hokkaido_snow_new_json"
    for img_oss in new_img_osses:
        name = img_oss.split("/")[-1].split(".")[0]
        new_save_path = "%s/%s.json" % (save_path, name)
        info = dict()
        info["imgUrl"] = img_oss
        with open(new_save_path, 'w') as output_file:
            json.dump(info, output_file)
    print(save_path)    
    
    # new_img_oss_base = new_img_osses[0].replace(new_img_osses[0].split("/")[-1],"")
    # file_list = []
    # total_file_len = len(open(args.txt_path,"r").readlines())
    # for oss_path in tqdm(open(args.txt_path,"r"),total = total_file_len):
    #     oss_path = oss_path.replace("\n","")
    #     with open(oss_path, 'r', encoding='utf-8') as f:
    #         json_file = json.load(f)
    #         if(new_img_oss_base+json_file["hash_image_name"] in new_img_osses):
    #             json_file["imgUrl_fake"] = new_img_oss_base+json_file["hash_image_name"]
    #             json_file["is_fake_image"] = True
    #             json_file["fake_name"] = args.fake_name
    #             json_file = json.dumps(json_file)
    #             file_list.append(json_file)
    # print("===========label card:===================")
    # card_engine.create_card_oss(card_files = file_list,project = args.project_name,media_name = args.media_name)

    

if __name__ == "__main__":
    main()

#python GAN_inf_2_train_card.py --txt_path /share/kevin/GANs/data_path/day_8w_add_tag.txt --infer_path /mnt/share_disk/kevin/STROTSS/8W/ --media_name fake_night --fake_name fake_night

#python GAN_inf_2_train_card.py --txt_path /data_path/gans_10w/base5.txt --infer_path /mnt/share_disk/kevin/STROTSS/8W/ --media_name fake_night --fake_name fake_night

#python GAN_inf_2_train_card.py --txt_path /share/kevin/GANs/LANE/dataset/samples/day_sample_with_tag_remove_empty_4.txt --infer_path /mnt/share_disk/kevin/LANE_FRONT/inf/BDD/ --media_name lane --fake_name fake_night

#python GAN_inf_2_train_card.py --txt_path /share/kevin/GANs/LANE/dataset/samples/day_sample_with_tag_remove_empty_3.txt --infer_path /mnt/share_disk/kevin/LANE_FRONT/inf/BASE_4_remain/ --media_name lane --fake_name fake_night

#python GAN_inf_2_train_card.py --txt_path /share/kevin/GANs/LANE/dataset/samples/day_sample_5w_with_tag_remove_empty.txt --infer_path /root/5W_LANE_FRONT_2/inf/BASE_4/ --media_name lane --fake_name fake_night


