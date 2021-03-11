from conf import conf

from dataclasses import replace, dataclass, field

import tensorflow_datasets as tfds
import tensorflow as tf


@dataclass
class Dataset:
    name: str
    split: str or list
    NUM_CLASSES: int = field(init=False)
    ds: tf.data.Dataset = field(init=False)
    ds_train: tf.data.Dataset = field(init=False)
    ds_val: tf.data.Dataset = field(init=False)
    ds_test: tf.data.Dataset = field(init=False)

    def load(self):
        raise NotImplementedError


@dataclass
class _Cifar(Dataset):

    def load(self):
        # in distribution data
        (self.ds_train, self.ds_val, self.ds_test), ds_info = tfds.load(
            self.name, split=self.split, with_info=True, as_supervised=True
        )
        self.NUM_CLASSES = ds_info.features['label'].num_classes


@dataclass
class Cifar10(_Cifar):
    name = 'cifar10'
    split = ['train[:80%]', 'train[80%:]', 'test']


@dataclass
class Cifar100(_Cifar):
    name = 'cifar100'
    split = ['train[:80%]', 'train[80%:]', 'test']


@dataclass
class SVHNCropped(Dataset):
    name = 'svhn_cropped'
    split = 'test'

    def load(self):
        ood_data, ds_info = tfds.load(self.name, split=self.split, with_info=True, as_supervised=False)

        # extract image and label from dictionary
        ood_data = ood_data.map(lambda d: (d['image'], d['label']))
        self.NUM_CLASSES = ds_info.features['label'].num_classes

        # take 1000 instances from each label
        ood_data = [ood_data.filter(lambda img, lbl: lbl == i) for i in range(self.NUM_CLASSES)]
        ood_data = list(map(lambda ds: ds.take(1000), ood_data))

        # concatenate all samples from labels
        ds = ood_data[0]
        for d in ood_data[1:]:
            ds = ds.concatenate(d)

        self.ds = ds


@dataclass
class Textures(Dataset):
    name = 'dtd'
    split = 'train+validation+test'

    def load(self):
        ood_data, ds_info = tfds.load(self.name, split=self.split, with_info=True, as_supervised=False)

        ood_data = ood_data.map(lambda d: (d['image'], d['label']))

        self.ds = ood_data

