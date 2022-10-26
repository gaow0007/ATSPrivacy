export CUDA_VISIBLE_DEVICES=$1
data=cifar100
epoch=10
python -u benchmark/cifar100_train.py --data=$data --arch=vit --epochs=$epoch --aug_list='' --mode=crop
