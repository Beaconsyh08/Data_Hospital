# -*- coding:utf-8 -*-

import json
import requests
import random
import time


def test_evaluate_request():
    data = {"request_id": "0811_sidecam_113w_v2.2.8.1",
            "project_name": "icu30",
            "algorithm_type": "perception",
            "model_type": "model",
            "model_name": "2d_obstacle",
            "sensor": "front_left_camera",
            "tasks": [
                {"gt_card": {"project": "icu30", "id": "62d173f328f4f2a8671b2874", "name": "trans_zhouping_side_gt"},
                 "dt_card": {"project": "icu30", "id": "62f4ff0a30428c644bf4dc39", "name": "2281_SIDECAM"}}]
                #  {"gt_card": {"project": "icu30", "id": "629f1c2c4db89afdaca01bd4", "name": ""},
                #  "dt_card": {"project": "logsim_perception_test", "id": "62ac3115686f561aac44f6c4", "name": ""}}]
            }

    r = requests.post(url="http://10.100.4.203:5000/evaluate_lucas", json=data)
    print(r.json())


def test_get_request():
    data = {"request_id": "0811_sidecam_113w_v2.2.8.1"}

    r = requests.post(url="http://10.100.4.203:5000/get_lucas_evaluate_result", json=data)
    print(r.json())


if __name__ == "__main__":
    # test_evaluate_request()
    test_get_request()