cd /home/siso/projects/siso/vicente

now=`date +"%Y-%m-%d_%H:%M:%S"`

#python3 video_test.py -m1 ../models/model_inception_siso_with_nthu_10k_with_2class.h5 -m2 ../models/model_yawn.h5 -o /home/siso/projects/siso/output/ -w 0 > /home/siso/projects/siso/output/siso_log_${now}.txt
python3 video_test.py -m1 ../models/model_inception_siso_with_nthu_10k_with_2class.h5 -m2 ../models/model_yawn.h5 -o /home/siso/projects/siso/output/ -w 0 
