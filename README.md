# Privacy-preserving Collaborative Learning with Automatic Transformation Search

This repository is an implementation of the privacy-preserving algorithm discussed in
```
Privacy-preserving Collaborative Learning with Automatic Transformation Search
```
which can be found at https://arxiv.org/abs/2011.12505
<p align="center">
    <img src="assets/framework.png" width="80%" />
</p>


### Abstract
Collaborative learning has gained great popularity due to its benefit of data privacy protection: participants can jointly train a Deep Learning model without sharing their training sets. However, recent works discovered that an adversary can fully recover the sensitive training samples from the shared gradients. Such reconstruction attacks pose severe threats to collaborative learning. Hence, effective mitigation solutions are urgently desired.
In this paper, we propose to leverage data augmentation to defeat reconstruction attacks: by preprocessing sensitive images with carefully-selected transformation policies, it becomes infeasible for the adversary to extract any useful information from the corresponding gradients. We design a novel search method to automatically discover qualified policies. We adopt two new metrics to quantify the impacts of transformations on data privacy and model usability, which can significantly accelerate the search speed. Comprehensive evaluations demonstrate that the policies discovered by our method can defeat existing reconstruction attacks in collaborative learning, with high efficiency and negligible impact on the model performance.

## Code
The search algorithm file that contains  can be found at ```searchalg/```. The other important experimental part can be found at ```benchmark/```.

## Setup
You can use [anaconda](https://www.anaconda.com/distribution/) to install our setup by running
```
conda env create -f environment.yml
conda activate ats
```


## Quick Start 
### Step1 Download Key Files 
[tiny_data trained network for search](https://entuedu-my.sharepoint.com/:u:/g/personal/gaow0007_e_ntu_edu_sg/ETkgPi-c17pAtR2ykbGCI8cBoK0651EgFIeBTdJVSW99Zw?e=FIpq6E)


### Step2 Automatic Transformation Search
```
# download a tiny_data trained network for search 
python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100 > batch_generate.sh
bash batch_generate.sh
python -u searchalg/search_best.py --arch=ResNet20-4 --data=cifar100
```

### Step3 Privacy-Perserving Evaluation
```
bash benchmark/run/benchmark_attack.sh
```

### Visualize Result
Input Image             |  Reconstruction from Inverting Gradients
:-------------------------:|:-------------------------:
![](assets/normal_ori.png)  |  ![](assets/normal_rec.png)

Transformed Image             |  Reconstruction from Inverting Gradients
:-------------------------:|:-------------------------:
![](assets/defense_ori.png)  |  ![](assets/defense_rec.png)

[Model: standard ResNet20, trained on cifar100 data. The image is from the validation set.]


# Acknowledgement 
We express gratitudes to the great work [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) and [DLG](https://github.com/mit-han-lab/dlg) as we benefit a lot from both their papers and codes.

# License
This repository is released under the MIT license. 
