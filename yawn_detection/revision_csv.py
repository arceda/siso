# este script se creo para crear un csv que contenga la revision de la imagenes de la BD de yawning, 
# el la revision se separaron en la yawning_incorrect, las imagenes mal anotadas
# este script solo crea un csv con el contenido de la muestra de yawninig y las yawning mal anotadas

import glob
import numpy as np
import pandas as pd
import sys

path_db = "/home/vicente/datasets/NTHU_IMG/4_classes_40k/"
path_db = sys.argv[1]

files_yawn = glob.glob(path_db + "yawning/*")
files_yawn_incorrect = glob.glob(path_db + "yawning_incorrect/*")

data = []
for file in files_yawn:
    file_name = file.split("/")[-1]
    if "(copy)" in file:
        pass
    else:
        data.append([file_name, "ok"])

for file in files_yawn_incorrect:
    file_name = file.split("/")[-1]
    if "(copy)" in file:
        pass
    else:
        data.append([file_name, "error"])

data = np.array(data)

data_frame = pd.DataFrame(data=data, columns=['file', 'status'])
print(data_frame)

data_frame.to_csv("yawning_revision.csv")