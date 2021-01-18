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

# while [ $i -le $1 ]
# do
#   #echo "ran"
#   #extract patches for alert subjects
#   mkdir Output/$i
#   mkdir Output/$i/Alert
#   echo "Alert $i"
#   python eyedetection.py 1 Input/$i/testAlert$i.avi ./ Output/$i/Alert/
#   #python eyedetection.py 1 Input/$i/ ./ $i/Alert/
#   #make dir i,i/alert,i/sleepy
#   #python eyedetection.py 1
#   i=$(($i + 1))
# done
# #python eyedetection.py 1 Input/$i.avi ./ $i/Alert/
# i=1
# #i=$(($i + 1))
# #echo $i
# #mkdir $i
# while [ $i -le $1 ]
# do
#   #echo "ran"
#   #extract patches for drowsy subjects
#   mkdir Output/$i/Drowsy
#   echo "Drowsy $i"
#   python eyedetection.py 1 Input/$i/testDrowsy$i.avi ./ Output/$i/Drowsy/
#   #make dir i,i/alert,i/sleepy
#   #python eyedetection.py 1
#   i=$(($i + 1))
# done


while [ $i -le $1 ]
do
  #echo "ran"
  #extraer cortes para sujetos somnolientos
  mkdir Output/$i
  echo "Test $i"
  python eyedetection.py 1 "Input/$i/test ($i).mp4" ./ Output/$i/
  #make dir i,i/alert,i/sleepy
  #python eyedetection.py 1
  i=$(($i + 1))
done

# while [ $i -lt 10 ]
# do
#   echo "ran"
#   make dir i,i/alert,i/sleepy
#   python eyedetection.py 1
#   i=$(($i + 1))
# done
#python eyedetection.py
