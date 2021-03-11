from dataclasses import dataclass, asdict, field
import json

from util import save_import_tensorflow
tf = save_import_tensorflow(gpu='2')

from make_datasets import Dataset, Cifar10, Textures, train_split


@dataclass
class Configuration:
    in_distribution_data: Dataset = Cifar10(train_split)
    out_of_distribution_data: Dataset = Textures()
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 128
    EPOCHS: int = 20
    loss = tf.losses.CategoricalCrossentropy(label_smoothing=0.2)
    loss_repr: dict = field(default_factory=lambda: {})
    NUM_WORKER: int = tf.data.experimental.AUTOTUNE
    MODEL_FUNCTION: str = 'effnetb0_custom_build_model'

    def __post_init__(self):

        if not hasattr(self, 'loss_repr'):
            self.loss_repr = self.loss.get_config()

        self.checkpoint_filepath = './trained_models/checkpoint/' + self.in_distribution_data.name + '/'

    def make_model(self) -> tf.keras.Model:
        import models
        m_fun = getattr(models, self.MODEL_FUNCTION)
        return m_fun(self.IMG_SIZE, self.in_distribution_data.NUM_CLASSES, self.loss)

    def save(self):
        with open(self.checkpoint_filepath + 'conf.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

    @classmethod
    def LoadConfig(cls, in_distribution_data: str):
        checkpoint_filepath = './trained_models/checkpoint/' + in_distribution_data + '/'
        with open(checkpoint_filepath + 'conf.json', 'r', encoding='utf-8') as f:
            r = json.load(f)

        r = cls(**r)
        r.loss = 'Cannot recover callable losses from LoadConfig.'
        return r


conf = Configuration()
