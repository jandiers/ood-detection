from dataclasses import replace

from util import save_import_tensorflow
tf = save_import_tensorflow('1')

from make_datasets import Dataset, val_split, test_split, Cifar10, SVHNCropped, Food101
from label_transformations import OneHotLabelTransformer
from sklearn import metrics, ensemble
import numpy as np
import pandas as pd


ds = Cifar10(test_split)
ood = SVHNCropped()


def get_adversarial_predictions(data, model, epsilon: float = 0.3):
    softmax = tf.keras.layers.Softmax()
    criterion = tf.keras.losses.categorical_crossentropy

    prd = []
    for x, y, _ in data:
        with tf.GradientTape() as t:
            t.watch(x)
            output = softmax(model(x))
            loss = criterion(y, output)

        gradients = t.gradient(loss, x)
        gradients = tf.sign(gradients)

        x_tilde = x - epsilon * gradients
        p = model.predict(x_tilde)
        prd.append(p)

    prd = np.hstack(prd)

    return prd


def fpr95(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ix = np.argwhere(tpr >= 0.95).ravel()[0]
    return fpr[ix]


def odin_results(ds: Dataset, ood: Dataset):
    from conf import conf, normal

    ds = replace(ds, label_transformer=OneHotLabelTransformer(num_classes=ds.NUM_CLASSES))
    ood = replace(ood, label_transformer=OneHotLabelTransformer(num_classes=ds.NUM_CLASSES))

    ds_id_val = replace(ds, split=val_split, label_transformer=OneHotLabelTransformer(num_classes=ds.NUM_CLASSES))
    ds_ood_val = Food101(val_split, label_transformer=OneHotLabelTransformer(num_classes=ds.NUM_CLASSES))

    ds_id_val = ds_id_val.load()
    ds_ood_val = ds_ood_val.load()

    result_collection = dict()

    # load model
    conf = replace(conf, strategy=normal, in_distribution_data=ds, out_of_distribution_data=None)
    model = conf.make_model()
    model.load_weights(conf.checkpoint_filepath)

    # get logits before softmax activation
    o = model.get_layer(name='pred')
    o.activation = None
    model = tf.keras.Model(model.input, o.output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(f'ODIN: {ds.__class__.__name__} vs. {ood.__class__.__name__}')

    pred_ds = get_adversarial_predictions(ds_id_val, model, epsilon=0.3)
    pred_ood = get_adversarial_predictions(ds_ood_val, model, epsilon=0.3)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

    # temperature scaling
    Ts = [1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.][::-1]
    softmax = tf.keras.layers.Softmax()

    for t in Ts:   # test different Ts for temperature scaling
        p = pred / t
        p = softmax(p).numpy()
        prec = 0.

        for delta in np.linspace(0.2, 0.8, 40):  # test different deltas for minimum confidence of predictions
            # check precision
            ood_p = p.max(axis=1) > delta
            ood_p = ood_p.astype(int)
            prec = metrics.accuracy_score(ood_labels, ood_p)

            if prec > 0.95:
                print('delta found:', delta.round(3), 'T:', t, 'precision:', prec.round(3))
                break

        if prec > 0.95:
            # take highest T if 95% precision is reached
            break

    ds = ds.load()
    ood = ood.load()

    pred_ds = get_adversarial_predictions(ds, model, epsilon=0.3)
    pred_ood = get_adversarial_predictions(ood, model, epsilon=0.3)

    y_true = np.hstack([y.numpy() for (x, y, w) in ds.load()])
    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

    # temperature scaling
    pred = pred / t   # t is optimized in loop before
    pred = softmax(pred).numpy()

    # OOD accuracies
    all_pred = (pred > delta).any(1).astype(int)   # 1 if classified as in distribution

    ood_error = 1. - metrics.accuracy_score(ood_labels, all_pred)
    print('OOD Error:', ood_error)
    result_collection['OOD error'] = ood_error

    p = pred.max(1)
    ood_auc = metrics.roc_auc_score(ood_labels, p)
    result_collection['OOD AUC'] = ood_auc
    print('OOD Area under Curve:', ood_auc)

    fpr = fpr95(ood_labels, p)
    result_collection['FPR at 95% TPR'] = fpr
    print('FPR at 95% TPR:', fpr)

    return result_collection
