from dataclasses import dataclass, field, make_dataclass, replace
from typing import NewType, Union
import re

from label_transformations import LabelTransformer
from models import augment_image
from util import save_import_tensorflow

tf = save_import_tensorflow(gpu='3')
import tensorflow_datasets as tfds

Split = NewType('Split', str)
train_split: Split = Split('train')
val_split: Split = Split('val')
test_split: Split = Split('test')

"""
Classes for dataset loading
"""


@dataclass
class Dataset:
    split: Split

    IMG_SIZE: int = 224
    BATCH_SIZE: int = 128
    NUM_WORKER: int = 4  # tf.data.experimental.AUTOTUNE
    CACHE: bool = False

    num_samples: int = field(default=None)
    sample_weight: float = field(default=None)
    label_transformer: LabelTransformer = field(default=None)
    NUM_CLASSES: int = field(default='not_yet_loaded', init=False)
    ds: tf.data.Dataset = field(default='not_yet_loaded', init=False)
    name: str = field(init=False, repr=False)
    is_train_set: bool = field(init=False, repr=True)

    def __post_init__(self):

        self.is_train_set = bool(re.match(r'train$|train\[:\d+%]', self.split))

        self._load()
        if self.num_samples:
            self.ds = self.ds.take(self.num_samples)

        if self.label_transformer is not None:
            if self.label_transformer.num_classes is None:
                self.label_transformer = replace(self.label_transformer, num_classes=self.NUM_CLASSES)
            else:
                self.NUM_CLASSES = self.label_transformer.num_classes

            self.ds = self.ds.map(self.label_transformer)

        self.sample_weight = self.sample_weight or 1.
        self.ds = self.ds.map(lambda x, y: (x, y, self.sample_weight))

        self.ds = self._prepare_data(self.ds, cache=self.CACHE)

    def load(self):
        return self.ds

    def _load(self):
        raise NotImplementedError

    def _prepare_data(self, ds, cache: bool = False):

        size = (self.IMG_SIZE, self.IMG_SIZE)
        ds = ds.map(lambda image, label, weight: (tf.image.resize(image, size), label, weight))

        if cache:
            ds = ds.cache()

        if self.is_train_set:
            ds = ds.map(augment_image, num_parallel_calls=self.NUM_WORKER)

        if self.is_train_set:
            ds = ds.shuffle(5000)

        ds = ds.batch(self.BATCH_SIZE)
        ds = ds.prefetch(self.NUM_WORKER)

        return ds


@dataclass
class _Cifar(Dataset):
    def _load(self):
        # in distribution data
        self.ds, ds_info = tfds.load(
            self.name, split=self.split, with_info=True, as_supervised=True
        )
        self.NUM_CLASSES = ds_info.features['label'].num_classes

    def __post_init__(self):
        if self.split == train_split:
            self.split = Split('train[:80%]')
        if self.split == val_split:
            self.split = Split('train[80%:]')
        super(_Cifar, self).__post_init__()


Cifar10 = make_dataclass('Cifar10', [
    ('split', Split),
    ('num_samples', int, None),
    ('name', str, 'cifar10')], bases=(_Cifar,))


Cifar100 = make_dataclass('Cifar100', [
    ('split', Split),
    ('num_samples', int, None),
    ('name', str, 'cifar100')], bases=(_Cifar,))


@dataclass
class SVHNCropped(Dataset):
    split: Split = Split('test')
    name: str = 'svhn_cropped'

    def _load(self):
        ood_data, ds_info = tfds.load(self.name, split=self.split, with_info=True, as_supervised=False)
        self.NUM_CLASSES = ds_info.features['label'].num_classes

        # extract image and label from dictionary
        ood_data = ood_data.map(lambda d: (d['image'], d['label']))

        # take 1000 instances from each label
        ood_data = [ood_data.filter(lambda img, lbl: lbl == i) for i in range(self.NUM_CLASSES)]
        ood_data = list(map(lambda ds: ds.take(1000), ood_data))

        # concatenate all samples from labels
        ds = ood_data[0]
        for d in ood_data[1:]:
            ds = ds.concatenate(d)

        self.ds = ds


@dataclass
class _LoadFromDictDataset(Dataset):
    def _load(self):
        ood_data, ds_info = tfds.load(self.name, split=self.split, with_info=True, as_supervised=False)
        self.NUM_CLASSES = ds_info.features['label'].num_classes

        ood_data = ood_data.map(lambda d: (d['image'], d['label']))

        self.ds = ood_data


@dataclass
class Textures(_LoadFromDictDataset):
    split: Split = Split('train+validation+test')
    name: str = 'dtd'


@dataclass
class Food101(_LoadFromDictDataset):
    split: Split = Split('train')
    name: str = 'food101'

    def __post_init__(self):
        if self.split == train_split:
            self.split = Split('train[:80%]')
        if self.split == val_split:
            self.split = Split('train[80%:]')
        if self.split == test_split:
            self.split = Split('validation')

        super(Food101, self).__post_init__()
