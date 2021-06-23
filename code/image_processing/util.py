"""
Utils.
"""

import os
import cv2


def print_right(text):
    print("\033[1;32m" + text +"\033[0m")

def simple_data_loader(data_dir, num=10000):
    file_name_list = sorted(os.listdir(data_dir))
    img_list = []
    # for all files in data_dir
    for count, file_name in enumerate(file_name_list):
        # only load 'num' images
        if count == num:
            break
        # only load .jpg and .png
        if file_name[0] == '.' or \
                (file_name.split('.')[-1] != 'jpg' and file_name.split('.')[-1] != 'png'):
            continue
        img = cv2.imread(os.path.join(data_dir, file_name))
        # resize
        h, w = img.shape[:2]
        img = cv2.resize(img, (512, int(512*h/w)))
        # append
        img_list.append(img)
    return img_list
