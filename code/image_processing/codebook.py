import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans

from util import *

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../../data"
DATA_DIR = "ProblemC_Files"
IS_COMPUTE_DESCRIPTOR = False
IS_BUILD_CODEBOOK = False
IS_COMPUTE_DISTANCE = True

K = 32
BATCH_SIZE = 100


def SIFT_extractor(img_list):
    img_key_list, img_des_list = [], []
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=300)
    for img in img_list:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        img_key_list.append(keypoints)
        img_des_list.append(descriptors)
    return img_key_list, img_des_list


def codebook_builder(descriptors, k, batch_size):
    X = np.vstack((des for des in descriptors))  # X{array-like, sparse matrix} of shape (n_samples, n_features)
    # k_means = KMeans(n_clusters=k)
    k_means = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    k_means.fit(X)
    codebook = k_means.cluster_centers_.squeeze()
    return codebook


if __name__ == "__main__":

    if IS_COMPUTE_DESCRIPTOR:
        # Load data
        print_right("Loading data...")
        img_list = simple_data_loader(os.path.join(DATA_PATH, DATA_DIR))

        # SIFT extracting
        print_right("Extracting SIFT descriptor...")
        img_key_list, img_des_list = SIFT_extractor(img_list)

        # File operations
        with open('../../data/pkl/'+DATA_DIR+'_SIFT.pkl', 'wb') as f:
            pickle.dump(img_des_list, f)

    else:
        # Load descriptors
        print_right("[Pickle] loading descriptors...")
        with open('../../data/pkl/'+DATA_DIR+'_SIFT.pkl', 'rb') as f:
            img_des_list = pickle.load(f)
        print("img_des_list len:", len(img_des_list))

    if IS_BUILD_CODEBOOK:
        # Build codebook
        print_right("Building codebook...")
        codebook = codebook_builder(img_des_list, K, BATCH_SIZE)
        print("codebook shape:", codebook.shape)

        # Save codebook
        print_right("Saving codebook...")
        with open('../../data/pkl/codebook.pkl', 'wb') as f:
            pickle.dump(codebook, f)

    else:
        # Load codebook
        print_right("Loading codebook...")
        with open('../../data/pkl/codebook.pkl', 'rb') as f:
            codebook = pickle.load(f)
        print("codebook shape:", codebook.shape)

    if IS_COMPUTE_DISTANCE:
        # Compute distance
        print_right("Generating distance cube...")

        distance_cube = []
        for n in range(len(img_des_list)):  # n images
            dis_wrt_des = []
            for m in range(len(img_des_list[n])):   # m descriptors
                dis_wrt_feature = []
                for k in range(len(codebook)):
                    dis_in_128 = np.linalg.norm(img_des_list[n][m] - codebook[k])
                    dis_wrt_feature.append(dis_in_128)
                dis_wrt_des.append(dis_wrt_feature)
            if n % 500 == 0:
                print("Mat %d shape:" % n, np.array(dis_wrt_des).shape)
            distance_cube.append(dis_wrt_des)
        print("cube shape[0]:", len(distance_cube))

        # Save the distance result
        print_right("Saving the distance cube...")
        with open('../../data/pkl/distance_cube.pkl', 'wb') as f:
            pickle.dump(distance_cube, f)
