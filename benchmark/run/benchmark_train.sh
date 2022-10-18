export CUDA_VISIBLE_DEVICES=$1
data=cifar100
epoch=200
python benchmark/cifar100_train.py --data=$data --arch=ResNet20-4 --epochs=$epoch --aug_list='' --mode=crop
