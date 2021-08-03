# albu-MixMatch-pytorch

This is an unofficial implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) (NIPS 2019). Official code is [here](https://github.com/google-research/mixmatch) written by Google Research with Tensorflow. Paper summary is [here](https://smkim7.notion.site/MixMatch-A-Holistic-Approach-to-Semi-Supervised-Learning-300594207f4c47fe9d8b0f99e7eb9ead) written by myself.



### Requirements

You can easily install all requirements by the commend

```
pip install -r requirements.txt
```

 The biggest difference from the previous implementation is that it uses [Albumentations](https://github.com/albumentations-team/albumentations) for image augmentation instead of torchvision transforms. This can also be tuned by train configurations.

### Datasets

The code supports CIFAR-10, CIFAR-100, SVHN and STL-10 as mentioned in MixMatch paper. 

### Training

Train the model by 250 labeled data of CIFAR-10 dataset (default).

```
python train.py --num-labels 250 --datasets CIFAR10
```

The code includes different hyperparameters for config including

* alpha (default=0.75): Parameter for beta distribution during MixUp stage
* lambda_u (default=75): Weight between supervised and unsupervised loss
* T (default=0.5): Temperature during softmax sharpening for entropy minimization
* K (default=2): Number of augmentations for unsupervised images

Default all follows from the paper.

### Results

| CIFAR-10  |    250     |    500     |    1000    |    2000    |    4000    |
| :-------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|   Paper   | 88.92±0.87 | 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 |
| This code |   #TODO    |   #TODO    |   #TODO    |   #TODO    |   #TODO    |

### References

* [MixMatch-pytorch](https://github.com/YU1ut/MixMatch-pytorch) by YU1ut
* [Mixmatch-pytorch-SSL](https://github.com/Jeffkang-94/Mixmatch-pytorch-SSL) by Jeffkang-94
* [mixmatch](https://github.com/google-research/mixmatch) by google-research

```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```

