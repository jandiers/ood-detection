from dataclasses import dataclass, field
from typing import NewType
from util import save_import_tensorflow
tf = save_import_tensorflow(gpu='1')

TransformedLabels = NewType('TransformedLabels', callable)


"""
Classes for Label Transformations
"""


@dataclass
class LabelTransformer:

    num_classes: int = field(default=None)

    def __call__(self, x, y) -> TransformedLabels:
        raise NotImplementedError


@dataclass
class UniformLabelTransformer(LabelTransformer):
    @tf.autograph.experimental.do_not_convert
    def __call__(self, x, y) -> TransformedLabels('uniform'):
        uniform_value = 1. / self.num_classes
        y = tf.ones(self.num_classes) * uniform_value
        return x, y


@dataclass
class OneHotLabelTransformer(LabelTransformer):
    @tf.autograph.experimental.do_not_convert
    def __call__(self, x, y) -> TransformedLabels('onehot'):
        return x, tf.one_hot(y, depth=self.num_classes)
