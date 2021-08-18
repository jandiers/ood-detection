import tensorflow as tf

layers = tf.keras.layers


def ood_accuracy(y_true, y_pred):
    # cast uniform labels to 0/1 indicators if ood or not: [0.33, 0.33, 0.33] => [0]; [0., 0., 1.,] => [1]
    y_true = tf.reduce_max(y_true, axis=-1)
    y_true = tf.equal(y_true, tf.constant([1.]))
    y_true = tf.cast(y_true, dtype=tf.int32)

    # cast predictions to predicted class or ood-input
    y_pred = tf.reduce_max(y_pred, axis=-1)
    y_pred = tf.greater_equal(y_pred, tf.constant(0.5))
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def effnetb0_custom_build_model(img_size, num_classes, loss):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    # x = img_augmentation(inputs)
    model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

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


def augment_image(image, label, weight):
    p = 0.5

    # left right flip
    if tf.random.uniform([]) < p:
        image = tf.image.flip_left_right(image)

    # adjust brightness
    if tf.random.uniform([]) < p:
        image = tf.image.adjust_brightness(image, 0.1)

    return image, label, weight
