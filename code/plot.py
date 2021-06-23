import matplotlib.pyplot as plt
import numpy as np
import os

BATCH_SIZE = 32
LR = "0001"
EPOCH = 160
EVAL_FREQ = 1
EVAL_EPOCH = EPOCH // EVAL_FREQ
NAME = "180e"

PATH = "../data/"+NAME+".log"
COLOR = ["red", "blue", "green", "cyan"]

def ploting(epoch, data, xlabel, ylabel, label=None, mode=None, lim=None):
    # Constructing x
    x = range(0, epoch, EVAL_FREQ)
    x = list(x)
    # Plot
    for i in range(len(data)):
        plt.plot(x, data[i], color=COLOR[i], linewidth=1, linestyle='-', label=label[i])
        if lim is not None:
            plt.ylim(lim[0], lim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(mode + ' - ' + str(epoch) + "epoch")
    plt.legend()
    # plt.show()
    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists("./result/"+NAME):
        os.mkdir("./result/"+NAME)
    plt.savefig(
        "./result/" + NAME + '/' + str(BATCH_SIZE) + 'B' + LR + 'LR' + str(epoch) + 'E' + mode + '.jpg', dpi=300)
    plt.close('all')


jianjiagufeiyechongdie_auc = []
weiying_auc = []
feiyebuquan_auc = []
suogubuping_auc = []
jianjiagufeiyechongdie_pixel_auc = []
weiying_pixel_auc = []
feiyebuquan_pixel_auc = []
suogubuping_pixel_auc = []

train_loss = []
val_loss = []

acc_top1 = []
acc_top2 = []
acc_top3 = []


def process_model(lines):
    epoch_counter = 0

    for i in range(len(lines)):
        words = lines[i].split(' ')
        if words[0] == '160':
            return
        if len(words) > 0:
            # train loss
            if words[0].replace('\n', '').replace('\r', '') == "--------------------------------" and epoch_counter < EVAL_EPOCH:
                train_loss.append(float(lines[i + 1].split(' ')[-1].replace('\n', '').replace('\r', '')))
                #  20 val 2021-02-06-18_10_24 val-loss: 0.0061 ||val-acc@1: 0.8537 val-acc@2: 0.9634 val-acc@3: 1.0000 ||time: 4
                val_line = lines[i + 6].split(' ')
                for j in range(len(val_line)):
                    if val_line[j] == 'val-loss:':
                        val_loss.append(float(val_line[j+1].replace('\n', '').replace('\r', '').replace('\t', '')))
                    if val_line[j] == '||val-acc@1:':
                        acc_top1.append(float(val_line[j+1].replace('\n', '').replace('\r', '').replace('\t', '')))
                    if val_line[j] == 'val-acc@2:':
                        acc_top2.append(float(val_line[j+1].replace('\n', '').replace('\r', '').replace('\t', '')))
                    if val_line[j] == 'val-acc@3:':
                        acc_top3.append(float(val_line[j+1].replace('\n', '').replace('\r', '').replace('\t', '')))
                i += 6
                epoch_counter += 1

    # Plot
    ploting(EPOCH, data=[train_loss, val_loss], xlabel="epoch", ylabel="loss",
            label=["train_loss", "val_loss"], mode="training loss & validation loss")
    ploting(EPOCH, data=[acc_top1, acc_top2, acc_top3], xlabel="epoch", ylabel="accuracy",
            label=["acc@top1", "acc@top2", "acc@top3"], mode="prediction accuracy")
    # ploting(EPOCH, data=[avg_pixel_auc, avg_val_loss/500], label=["pixel auc", "val loss"], mode="pixel auc & val loss", lim=(0, 1))
    # ploting(EPOCH, data=[avg_auc, avg_val_loss/500], label=["auc", "val loss"], mode="auc & val loss", lim=(0, 1))


if __name__ == '__main__':
    fo = open(PATH, "r")
    print("Processing:", fo.name)
    lines = fo.readlines()
    process_model(lines)

# "cls_top1": {"0": 0.8666666666666667, "2": 0.9166666666666666, "1": 0.95, "3": 0.8888888888888888, "4": 0.8823529411764706},
# "cls_top3": {"0": 1.0, "2": 0.9166666666666666, "1": 0.95, "3": 1.0, "4": 1.0},
