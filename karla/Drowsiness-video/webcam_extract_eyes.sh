#Input video name format:"test<Alert/Drowsy><test subject number>"
#Output video name format:"test<Alert/Drowsy><test subject number>"
i=1
#i=$(($i + 1))
#echo $i
#mkdir $i

#if Output exists, delete it
dir="Output"
if [ -d $dir ] ; then
    rm -rf $dir
fi

mkdir Output

#echo "ran"
#extract patches for drowsy subjects
echo "Test"
python end_to_end_new.py 1 "Input/3/test (3).mp4" ./ Output/
#make dir i,i/alert,i/sleepy
#python eyedetection.py 1
i=$(($i + 1))

# while [ $i -lt 10 ]
# do
#   echo "ran"
#   make dir i,i/alert,i/sleepy
#   python eyedetection.py 1
#   i=$(($i + 1))
# done
#python eyedetection.py
