# este script lee el csv e la revision de la bd yawning y muevbe  las imagenes mal etiquetadas a otra carpeta

import glob
import numpy as np
import pandas as pd
import sys
import os
from shutil import copyfile

path_db = "/home/vicente/datasets/NTHU_IMG/4_classes_40k/"
path_csv = "yawning_revision.csv"
path_db = sys.argv[1]
path_csv = sys.argv[2]

# python3 revision_csv_split.py "/home/vicente/datasets/NTHU_IMG/4_classes_4k/" "yawning_revision.csv"

if not os.path.exists(path_db + 'yawning_incorrect'):
    os.makedirs(path_db + 'yawning_incorrect')

data_frame = pd.read_csv(path_csv) 
data = data_frame.values
data = data[:, 1: 3]

for row in data:
    if row[1] == 'error':
        os.rename(path_db + "yawning/" + row[0], path_db + "yawning_incorrect/" + row[0])
