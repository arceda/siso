# this script convert format of videos to mp4 and scale them
# sudo apt-get install ffmpeg

#ffmpeg -i /home/vicente/datasets/SISO/normal.mp4    -vf scale=iw/2:-1  /home/vicente/datasets/SISO/normal_640_480.mp4

#ffmpeg -i /home/vicente/datasets/SISO/normal.mp4    -vf scale=640:480  /home/vicente/datasets/SISO/normal_640_480.mp4
#ffmpeg -i /home/vicente/datasets/SISO/normal2.mp4    -vf scale=640:480  /home/vicente/datasets/SISO/normal2_640_480.mp4
#ffmpeg -i /home/vicente/datasets/SISO/con_movimiento_ligero.mp4    -vf scale=640:480  /home/vicente/datasets/SISO/con_movimiento_ligero_640_480.mp4
#ffmpeg -i /home/vicente/datasets/SISO/con_movimiento.mp4    -vf scale=640:480  /home/vicente/datasets/SISO/con_movimiento_640_480.mp4
#ffmpeg -i /home/vicente/datasets/SISO/con_cubreboca.mp4    -vf scale=640:480  /home/vicente/datasets/SISO/con_cubreboca_640_480.mp4
#ffmpeg -i /home/vicente/datasets/SISO/con_cubreboca2.mp4    -vf scale=640:480   /home/vicente/datasets/SISO/con_cubreboca2_640_480.mp4

ffmpeg -i /home/vicente/datasets/SISO/nthu_normal_640_480.mp4 -vf scale=1920:1080  /home/vicente/datasets/SISO/nthu_normal.mp4