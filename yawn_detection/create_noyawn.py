# este script crea muestra de no yawning a partir de algunas imagenes de la nonspleepycombination

import glob
import numpy as np
import pandas as pd
import sys
from shutil import copyfile


path_db = "/home/vicente/datasets/NTHU_IMG/4_classes_40k/nonsleepyCombination/"
path_dest = "/home/vicente/datasets/NTHU_IMG/yawn/non_yawning/"
path_db = sys.argv[1]
path_dest = sys.argv[2]
# python3 create_noyawn.py "/home/vicente/datasets/NTHU_IMG/4_classes_40k/nonsleepyCombination/" "/home/vicente/datasets/NTHU_IMG/yawn/non_yawning/"
# python3 create_noyawn.py "/mnt/disk1/datasets/NTHU_IMG/4_classes_40k/FULL/nonsleepyCombination/" "/mnt/disk1/datasets/NTHU_IMG/yawn/non_yawning/"

files = glob.glob(path_db + "/*")

files = np.array(files)
np.random.shuffle(files)

print(files.shape)
files = files[0:4259]
print(files.shape)

for file in files:
    file_name = file.split("/")[-1]
    copyfile(file, path_dest + "/" + file_name)
