import numpy as np
import glop
from bitstring import BitArray
import matplotlib.pyplot as plt


angles = []

with open("test_timon_data.txt") as infile:
    for line in infile:
        str_line = line[1:len(line)]
        data = str_line.split(',')
        date = data[0]
        time = data[1]        
        signal_type = data[2]
        hard_correction = data[3]

        acc_pixels = data[4] # pixeles acumulados
        des_pixels = data[5] # pixeles desplazados
        cic_pixels = data[6] # pixeles por vuelta completa
        
        #print(date, time, signal_type)
        #print(acc_pixels, des_pixels, cic_pixels)
        acc_integer = BitArray('0x' + acc_pixels)
        des_integer = BitArray('0x' + des_pixels)
        cic_integer = BitArray('0x' + cic_pixels)

        #acc_integer.int -> signed int
        #acc_integer.uint -> unsigned int

        angle = acc_integer.int * 360 / cic_integer.int
        angles.append(angle)

        print(angle)


x = np.arange(0, len(angles), 1)
y = angles

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='sample', ylabel='angle',
       title='Angle')
ax.grid()

fig.savefig("angles.png")
plt.show()

