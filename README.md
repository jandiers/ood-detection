# Out-of-Distribution Detection using Outlier Detection Methods

Out-of-distribution detection (OOD) deals with anomalous input to neural networks. In the past, specialized methods have been proposed to identify anomalous input. Similarly, it was shown that feature extraction models in combination with outlier detection algorithms are well suited to detect anomalous input.  We use outlier detection algorithms to detect anomalous input as reliable as specialized methods from the field of OOD. No neural network adaptation is required; detection is based on the model's softmax score. Our approach works unsupervised using an Isolation Forest and can be further improved by using a supervised learning method such as Gradient Boosting.

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
Diers, J., Pigorsch, C. (2022). Out-of-Distribution Detection Using Outlier Detection Methods. In: Sclaroff, S., Distante, C., Leo, M., Farinella, G.M., Tombari, F. (eds) Image Analysis and Processing – ICIAP 2022. ICIAP 2022. Lecture Notes in Computer Science, vol 13233. Springer, Cham. https://doi.org/10.1007/978-3-031-06433-3_2
````
BibTeX:

````
@InProceedings{10.1007/978-3-031-06433-3_2,
author="Diers, Jan
and Pigorsch, Christian",
editor="Sclaroff, Stan
and Distante, Cosimo
and Leo, Marco
and Farinella, Giovanni M.
and Tombari, Federico",
title="Out-of-Distribution Detection Using Outlier Detection Methods",
booktitle="Image Analysis and Processing -- ICIAP 2022",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="15--26",
isbn="978-3-031-06433-3"
}


````
