from dataclasses import dataclass, asdict, field, replace
import json
from typing import NewType

from util import save_import_tensorflow

tf = save_import_tensorflow(gpu='1')

import foolbox as fb
from make_datasets import Dataset, Cifar10, Cifar100, Food101, Cars196, Cassava, train_split, val_split
from label_transformations import UniformLabelTransformer, OneHotLabelTransformer

TrainingStrategy = NewType('TrainingStrategy', str)
normal = TrainingStrategy('normal')
outlier_exposure = TrainingStrategy('outlier_exposure')

in_dist = Cifar100(train_split,
                   sample_weight=1.,
                   label_transformer=OneHotLabelTransformer())

ood = Food101(num_samples=len(in_dist.ds),
              sample_weight=0.5,
              label_transformer=UniformLabelTransformer(in_dist.NUM_CLASSES))

ood = None


@dataclass
class Configuration:
    strategy: TrainingStrategy = normal

    # in and out of distribution datasets
    in_distribution_data: Dataset = in_dist
    out_of_distribution_data: Dataset = ood

    # adversarial attack
    # attack: fb.Attack = fb.attacks.LinfPGD()

    # training details
    EPOCHS: int = 20
    loss = tf.losses.CategoricalCrossentropy(label_smoothing=0.2)
    loss_repr: dict = field(default_factory=lambda: {})
    MODEL_FUNCTION: str = 'effnetb0_custom_build_model'

    def __post_init__(self):
        self.val_ds = self._make_validation_set()

        if not hasattr(self, 'loss_repr'):
            self.loss_repr = self.loss.get_config()

        self.checkpoint_filepath = f'./trained_models/checkpoint/{self.strategy}/{self.in_distribution_data.name}/'

    def _make_validation_set(self):

        if (self.out_of_distribution_data is not None) \
                and (self.strategy == outlier_exposure):

            # validation set consists of 50% inlier and 50% outlier datasets
            in_dist_val = replace(self.in_distribution_data, split=val_split)
            ood_val = replace(self.out_of_distribution_data, split=val_split, num_samples=len(in_dist_val.ds))
            ds = tf.data.experimental.sample_from_datasets([in_dist_val.load(), ood_val.load()], [0.5, 0.5], seed=29)
            self.in_dist_val, self.ood_val = in_dist_val, ood_val
        else:
            # normal validation set
            ds = replace(self.in_distribution_data, split=val_split)
        return ds

    def make_model(self) -> tf.keras.Model:
        import models
        m_fun = getattr(models, self.MODEL_FUNCTION)
        return m_fun(self.in_distribution_data.IMG_SIZE, self.in_distribution_data.NUM_CLASSES, self.loss)

    def save(self):
        with open(self.checkpoint_filepath + 'conf.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

    @classmethod
    def LoadConfig(cls, in_distribution_data: Dataset):
        checkpoint_filepath = './trained_models/checkpoint/' + in_distribution_data.name + '/'
        with open(checkpoint_filepath + 'conf.json', 'r', encoding='utf-8') as f:
            r = json.load(f)

        r = cls(**r)
        r.loss = 'Cannot recover callable losses from LoadConfig.'
        return r


conf = Configuration()
