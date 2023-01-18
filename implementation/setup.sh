git clone https://github.com/AlexeyAB/darknet
cd darknet

sed -i '/GPU=0/c\GPU=1' ./Makefile
sed -i '/CUDNN=0/c\CUDNN=1' ./Makefile
sed -i '/OPENCV=0/c\OPENCV=1' ./Makefile

make 

cp cfg/yolov3.cfg cfg/yolov3-train.cfg

touch data/obj.names
touch data/obj.data

# TODO: write your own classes in same order as classes.txt
classes="cars drivers"
# TODO: change to number of classes you have
numClasses=2

for class in $classes
do
    echo "$class" >> data/obj.names
done

echo -e "classes = $numClasses\ntrain = data/train.txt\nvalid = data/test.txt\nnames = data/obj.names\nbackup = /content/implementation/pretrained/weight" > data/obj.data

mkdir data/train
cp -r ../implementation/data/train/* ./data/train

python3 ../implementation/setup.py --data_dir ./data/ --darknet_path .

wget https://pjreddie.com/media/files/darknet53.conv.74
 