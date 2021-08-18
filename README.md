# Out-of-Distribution Detection using Outlier Detection Methods

Out-of-distribution detection (OOD) deals with anomalous input to neural networks. In the past, specialized methods have 
been proposed to reject predictions on anomalous input. We use outlier detection algorithms to detect anomalous input 
as reliable as specialized methods from the field of OOD. No neural network adaptation is required; detection is based 
on the model's softmax score. Our approach works unsupervised with an Isolation Forest or with supervised classifiers 
such as a Gradient Boosting machine.

This repository contains the code used for the experiments, as well as instructions to reproduce our results. 

## Installation

This codebase mainly uses Tensorflow, scikit-learn and Tensorflow Datasets. The full details of the environment we 
used for our experiments may be found in `environment.yml`.


## Usage

If you want to use the code for your experiments, you first have to install the required packages.
Using a conda environment installed from environment.yml will be the easiest solution. You can also
check the environment.yml file to figure out which versions are supported in this code.

The datasets package provides convenience classes to load Tensorflow Datasets. For example, to load
the training split from Cifar10:

```python
from datasets.make_datasets import Cifar10, train_split
from datasets.label_transformations import OneHotLabelTransformer
cifar10 = Cifar10(train_split, label_transformer=OneHotLabelTransformer())
# Cifar10(split='train[:80%]', IMG_SIZE=224, BATCH_SIZE=32, NUM_WORKER=-1, ...)
ds = cifar10.load()
# ds is now an instance of tf.Dataset
```

Next, you need to train a model. For this, edit the `conf.py` file and run `train.py`. This will
train a model as described in our paper.

Once the training is finished, you can run the evaluation scripts. The `run_evalation.py` will 
evaluate all methods that we use in the paper.

More details may be found in our publication. If you use this work for your publication, we kindly ask
to cite our work. Also, please consider to cite the repositories from 
[Chen et al.](https://github.com/jfc43/robust-ood-detection) and 
[Liang et al.](https://github.com/facebookresearch/odin) that our code relies on. 

````
The citation will appear here soon.
````


````
BibTeX yet to come.
````
