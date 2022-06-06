import numpy as np
import os, sys
import argparse


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--alpha', default=0.7, type=float, help='Privacy score weight')

opt = parser.parse_args()
def normalize(score):
    range = np.max(score)-np.min(score)
    score = (score-np.min(score))/range
    return score

if __name__ == '__main__':
    search_root = 'search/data_{}_arch_{}/'.format(opt.data, opt.arch)
    acc_root = 'accuracy/data_{}_arch_{}/'.format(opt.data, opt.arch)
    maxpath, maxval = None, -sys.maxsize
    minpath, minval = None, sys.maxsize

    pri_scores_list = list()
    pri_vars_list = list()
    results = list()
    for path in os.listdir(acc_root):
        pathname = os.path.join(search_root, path)
        pri_score_list = np.load(pathname).tolist()
        pri_mean = np.mean(pri_score_list)
        pri_var = np.var(pri_score_list)

        pri_scores_list.append(pri_mean)
        pri_vars_list.append(pri_var)

        pathname = os.path.join(acc_root, path)
        acc_score_list = np.load(pathname).tolist()
        acc_mean = np.mean(acc_score_list)
        if path == '.npy':
            print(pri_mean, acc_mean)
        results.append((path, pri_mean, acc_mean))
        
    pri_scores_list, pri_vars_list = normalize(pri_scores_list), normalize(pri_vars_list)
    pri_score_weight_sum = opt.alpha*pri_scores_list + (1-opt.alpha)*pri_vars_list

    indices = np.argsort(pri_score_weight_sum)
    for i, idx in enumerate(indices):
        if results[idx][2] >= −85:
            print(f'Policy:{results[idx][0]}, Privacy Score: {pri_score_weight_sum[idx]}, Accuracy Score: {results[idx][2]}')


