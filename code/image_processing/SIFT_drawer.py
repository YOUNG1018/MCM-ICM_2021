import numpy as np
from util import *

DATAPATH = "../../data"


def SIFT_extractor(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=300)
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=400)
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    annotated_img = cv2.drawKeypoints(gray_img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return annotated_img


img_list = simple_data_loader(os.path.join(DATAPATH, "Asian giant hornet"))
for index, img in enumerate(img_list):
    img_marked = SIFT_extractor(img)
    cv2.imwrite(os.path.join(DATAPATH, "keypoints", str(index)+"_sift_keypoints.jpg"), img_marked)
