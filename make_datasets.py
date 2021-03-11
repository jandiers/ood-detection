from dataclasses import dataclass, field
from typing import NewType
from util import save_import_tensorflow

tf = save_import_tensorflow(gpu='2')
import tensorflow_datasets as tfds

Split = NewType('Split', str)
train_split: Split = Split('train')
val_split: Split = Split('val')
test_split: Split = Split('test')

LoadingState = NewType('not_yet_loaded', str)


@dataclass
class Dataset:
    name: str
    split: Split
    NUM_CLASSES: int = field(default='not_yet_loaded', init=False)
    ds: tf.data.Dataset = field(default='not_yet_loaded', init=False)

    def __post_init__(self):
        self.load()

    def load(self):
        raise NotImplementedError


@dataclass
class _Cifar(Dataset):
    def load(self):
        # in distribution data
        self.ds, ds_info = tfds.load(
            self.name, split=self.split, with_info=True, as_supervised=True
        )
        self.NUM_CLASSES = ds_info.features['label'].num_classes


@dataclass
class Cifar10(_Cifar):
    split: Split
    name: str = field(default='cifar10', init=False)

    def __post_init__(self):
        if self.split == train_split:
            self.split = Split('train[:80%]')
        if self.split == val_split:
            self.split = Split('train[80%:]')
        super(Cifar10, self).__post_init__()


@dataclass
class Cifar100(_Cifar):
    split: Split
    name: str = field(default='cifar100', init=False)


@dataclass
class SVHNCropped(Dataset):
    split: Split = Split('test')
    name: str = field(default='svhn_cropped', init=False)

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
    split: Split = Split('train+validation+test')
    name: str = field(default='dtd', init=False)

    def load(self):
        ood_data, ds_info = tfds.load(self.name, split=self.split, with_info=True, as_supervised=False)

        ood_data = ood_data.map(lambda d: (d['image'], d['label']))

        self.ds = ood_data
