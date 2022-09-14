import sys
sys.path.append("../haomo_ai_framework")
from haomoai.cards import cardapi
import random
from itertools import zip_longest


def card_generator(k: int, project: str, media_name: str, paths: list):

    def group_elements(n, iterable, padvalue=None):
        return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
            
    for path in paths:
        print(path)
        with open(path, "r") as ids:
            json_paths = [line.strip() for line in ids]

        for lst in group_elements(k, json_paths):
            clean_lst = [_ for _ in lst if _ is not None]
            print(len(clean_lst))
            clean_lst = [_[1:] for _ in clean_lst if _.startswith("/")]

            op = cardapi.CardOperation()
            card_id_train = op.create_card(clean_lst, project, media_name)
        
            print("Card ID: %s, Project: %s, Media Name: %s, Frame Amount: %d" % (str(card_id_train), str(project), str(media_name), len(clean_lst)))

PATHS = ['/data_path/v71_train_30W.txt']
K = 600000
PROJECT = "icu30"
MEDIA_NAME = "side"

card_generator(K, PROJECT, MEDIA_NAME, PATHS)
