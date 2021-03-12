# python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100 
import copy, random
import argparse

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
opt = parser.parse_args()


scheme_list = list()

num_per_gpu = 20



def write():
    for i in range(len(scheme_list) // num_per_gpu):
        print('{')
        for idx in range(i*num_per_gpu, i*num_per_gpu + num_per_gpu):
            sch_list = [str(sch) for sch in scheme_list[idx]]
            suf = '-'.join(sch_list)
             
            cmd = 'CUDA_VISIBLE_DEVICES={} python benchmark/search_transform_attack.py --aug_list={} --mode=aug --arch={} --data={} --epochs=100'.format(i%8, suf, opt.arch, opt.data)
            print(cmd)
        print('}&')



def backtracing(num, scheme):
    for _ in range(8 * 10 * num_per_gpu):
        scheme = list()
        for i in range(3):
            scheme.append(random.randint(-1, 50))
        new_policy = copy.deepcopy(scheme)
        for i in range(len(new_policy)):
            if -1 in new_policy:
                new_policy.remove(-1)
        scheme_list.append(new_policy)
    write()





if __name__ == '__main__':
    backtracing(5, scheme=list())
