from conf import conf, outlier_exposure
from models import ood_accuracy

from util import save_import_tensorflow
tf = save_import_tensorflow('1')
layers = tf.keras.layers
ReduceLROnPlateau = tf.python.keras.callbacks.ReduceLROnPlateau
EarlyStopping = tf.python.keras.callbacks.EarlyStopping

ds = conf.in_distribution_data.load()
ds_val = conf.val_ds

if conf.strategy == outlier_exposure:
    ds_out = conf.out_of_distribution_data.load()
    ds = tf.data.experimental.sample_from_datasets([ds, ds_out], [0.5, 0.5], seed=29)


model = conf.make_model()

hist = model.fit(ds, epochs=3, validation_data=ds_val)


def unfreeze_model(model, layer_name: str):
    beyond_layer = False
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if layer.name == layer_name:
            beyond_layer = True

        if not beyond_layer:
            continue

        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    if conf.strategy == outlier_exposure:
        metrics = ['accuracy', ood_accuracy]
    else:
        metrics = ['accuracy']

    model.compile(
        optimizer=optimizer, loss=conf.loss, metrics=metrics
    )


unfreeze_model(model, layer_name='block2b_add')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, min_delta=0.01,
                              patience=3, min_lr=1e-6, verbose=1)

early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=conf.checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

hist = model.fit(ds, epochs=conf.EPOCHS, initial_epoch=3, validation_data=ds_val,
                 callbacks=[reduce_lr,
                            early_stop,
                            model_checkpoint_callback])

model.evaluate(ds, verbose=1)

