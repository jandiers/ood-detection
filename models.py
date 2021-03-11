import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0


def effnetb0_custom_build_model(img_size, num_classes, loss):

    inputs = layers.Input(shape=(img_size, img_size, 3))
    # x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.Conv2D(320, 1, activation='relu', name='dim_red_conv')(model.output)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss=loss, metrics=['accuracy']
    )
    return model
