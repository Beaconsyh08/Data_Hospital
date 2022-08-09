# -*- coding:utf-8 -*-

import json
import requests
import random
import time


def test_evaluate_request():
    data = {"request_id": "20220719_sidecam-2",
            "project_name": "icu30",
            "algorithm_type": "perception",
            "model_type": "end_to_end",
            "model_name": "2d_lane_marking",
            "sensor": "front_middle_camera",
            "tasks": [
                {"gt_card": {"project": "qa", "id": "62d6289a0a4bcd0612b68b5c", "name": "gt"},
                 "dt_card": {"project": "qa", "id": "62d6290e0a4bcd0612b7cf3d", "name": "dt"}}]
                #  {"gt_card": {"project": "icu30", "id": "629f1c2c4db89afdaca01bd4", "name": ""},
                #  "dt_card": {"project": "logsim_perception_test", "id": "62ac3115686f561aac44f6c4", "name": ""}}]
            }

    r = requests.post(url="http://10.100.4.203:5000/evaluate_lucas", json=data)
    print(r.json())


def test_get_request():
    data = {"request_id": "0721-M-PT230-268-U"}

    r = requests.post(url="http://10.100.4.203:5000/get_lucas_evaluate_result", json=data)
    print(r.json())


if __name__ == "__main__":
    # test_evaluate_request()
    test_get_request()