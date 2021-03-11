from dataclasses import dataclass, asdict, field
import json


@dataclass
class Configuration:
    import gpu_utils
    import tensorflow as tf

    in_distribution_data: str = 'cifar100'
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 128
    EPOCHS: int = 20
    loss = tf.losses.CategoricalCrossentropy(label_smoothing=0.2)
    loss_repr: dict = field(default_factory=lambda: {})
    NUM_WORKER: int = tf.data.experimental.AUTOTUNE
    MODEL_FUNCTION: str = 'effnetb0_custom_build_model'

    def __post_init__(self):

        import tensorflow_datasets as tfds

        if not hasattr(self, 'loss_repr'):
            self.loss_repr = self.loss.get_config()

        self.checkpoint_filepath = './trained_models/checkpoint/' + self.in_distribution_data + '/'

        # in distribution data
        (self.ds_train, self.ds_val, self.ds_test), ds_info = tfds.load(
            self.in_distribution_data, split=['train[:80%]', 'train[80%:]', 'test'], with_info=True, as_supervised=True
        )
        self.NUM_CLASSES = ds_info.features['label'].num_classes

    def make_model(self) -> tf.keras.Model:
        import models
        m_fun = getattr(models, self.MODEL_FUNCTION)
        return m_fun(self.IMG_SIZE, self.NUM_CLASSES, self.loss)

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
