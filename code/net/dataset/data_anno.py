import os

DIR = "./test"

file_name_list = sorted(os.listdir(DIR))
f = open("test.txt", "w")

for file_name in file_name_list:

    # Asian_giant_hornet
    if file_name.split('_')[0] == "AGH":
        print("%s %d" % (file_name, 0), file=f)

    # Bee
    if file_name.split('_')[0] == "Bee":
        print("%s %d" % (file_name, 1), file=f)

    # Baldfaced hornets
    if file_name.split('_')[0] == "BH":
        print("%s %d" % (file_name, 2), file=f)

    # Eastern cicada killers
    if file_name.split('_')[0] == "ECK":
        print("%s %d" % (file_name, 3), file=f)

    # European hornets
    if file_name.split('_')[0] == "EH":
        print("%s %d" % (file_name, 4), file=f)
