import numpy as np
import os, sys
import argparse


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
opt = parser.parse_args()


if __name__ == '__main__':
    search_root = 'search/data_{}_arch_{}/'.format(opt.data, opt.arch)
    acc_root = 'accuracy/data_{}_arch_{}/'.format(opt.data, opt.arch)
    maxpath, maxval = None, -sys.maxsize
    minpath, minval = None, sys.maxsize

    results = list()
    for path in os.listdir(acc_root):
        pathname = os.path.join(search_root, path)
        content = np.load(pathname).tolist()
        val1 = np.mean(content)
        pathname = os.path.join(acc_root, path)
        content = np.load(pathname).tolist()
        val2 = np.mean(content)
        if path == '.npy':
            print(val1, val2)
        results.append((path, val1, val2))
        
    results.sort(key=lambda x: x[1])
    for idx, result in enumerate(results):
        if result[2] >= −85:
            print(result)


