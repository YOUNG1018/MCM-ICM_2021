import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import os
import cv2

POSITIVE = "Positive ID"
NEGATIVE = "Negative ID"
RED = 'r'
BLUE = 'b'

CSV_FILE = "../data/2021_MCM_Problem_C_Data/2021MCMProblemC_DataSet.csv"
OUT_PATH = "./figs"
FIG_NAME = "Map of Detections"


def draw(pos, y, pth):
    """
        Draw the scatter plot of given X(latitude & longitude) and y(Positive-red & Unverified/Negative-blue).
    """
    s1, s2 = None, None

    longitude = pos[0]
    latitude = pos[1]
    for i in range(len(y)):
        if y[i] == POSITIVE:
            s1 = plt.scatter(longitude[i], latitude[i], s=20, c=RED, alpha=0.2, label='Positive ID')
        elif y[i] == NEGATIVE:
            s2 = plt.scatter(longitude[i], latitude[i], s=20, c=BLUE, alpha=0.1, label='Negative ID')
        else:
            continue
    plt.title(FIG_NAME, fontsize=15, fontweight='bold')
    plt.xlabel("Longitude", fontsize=13, fontweight='bold')
    plt.ylabel("Latitude", fontsize=13, fontweight='bold')
    plt.xlim(-125, -116.5)
    plt.ylim(45.25, 49.75)
    plt.legend((s1, s2), (POSITIVE, NEGATIVE), loc='best')
    plt.savefig(pth + ".png", dpi=300, transparent=True)


if __name__ == "__main__":
    csv_data = pd.read_csv(CSV_FILE)

    position_x = csv_data["Longitude"].values.tolist()
    position_y = csv_data["Latitude"].values.tolist()

    lab_status = csv_data["Lab Status"].values.tolist()

    draw((position_x, position_y), lab_status, os.path.join(OUT_PATH, FIG_NAME))