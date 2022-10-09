
export CUDA_VISIBLE_DEVICES=0
data=vit
epoch=10
arch='vit'
# python benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=aug 
python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed'

# for aug_list in '3-1-7+43-18-18' '3-1-7' '43-18-18';
# do
# {
# echo $aug_list
# python -u benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug
# python -u benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug --optim='inversed'
# }&
# done