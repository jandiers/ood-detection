from dataclasses import replace

from util import save_import_tensorflow
tf = save_import_tensorflow('1')

from make_datasets import Dataset, val_split, test_split, Cifar10, SVHNCropped, Food101
from label_transformations import OneHotLabelTransformer
from sklearn import metrics, ensemble
import numpy as np
import pandas as pd
import tqdm


def fpr95(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ix = np.argwhere(tpr >= 0.95).ravel()[0]
    return fpr[ix]


def ODIN(ds, model, temperature, epsilon):

    # code ported from pytorch to tensorflow based on the implementations by Liang et al. (ODIN)
    # and Chen et al (robust out of distribution detection)

    criterion = tf.keras.losses.sparse_categorical_crossentropy

    ots = []
    for x, _, _ in tqdm.tqdm(ds):
        with tf.GradientTape() as t:
            t.watch(x)
            outputs = model(x)
            maxIndexTemp = tf.argmax(outputs, axis=1)
            outputs = outputs / temperature
            labels = tf.constant(maxIndexTemp)
            loss = criterion(labels, outputs, from_logits=True)

        gradients = t.gradient(loss, x)
        gradients = tf.sign(gradients)

        # Adding perturbations to images
        tempInputs = x - epsilon * gradients
        outputs = model.predict(tempInputs)
        outputs = outputs / temperature

        # Calculating the confidence after perturbations
        outputs = outputs - np.max(outputs, axis=1, keepdims=True)
        outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        ots.append(outputs)

    return np.vstack(ots)


def odin_results(ds: Dataset, ood: Dataset):
    from conf import conf, normal

    ds = replace(ds, BATCH_SIZE=16)
    ood = replace(ood, BATCH_SIZE=16)

    # load model
    conf = replace(conf, strategy=normal, in_distribution_data=ds, out_of_distribution_data=None)
    model = conf.make_model()
    model.load_weights(conf.checkpoint_filepath)

    # get logits before softmax activation for temperature scaling
    o = model.get_layer(name='pred')
    o.activation = None
    model = tf.keras.Model(model.input, o.output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(f'ODIN: {ds.__class__.__name__} vs. {ood.__class__.__name__}')
    result_collection = dict()
    ds = ds.load()
    ood = ood.load()

    pred_ds = ODIN(ds, model, temperature=2., epsilon=0.3)
    pred_ood = ODIN(ood, model, temperature=2., epsilon=0.3)

    y_true = np.hstack([y.numpy() for (x, y, w) in ds])
    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

    # OOD accuracies
    delta = 0.5
    all_pred = (pred > delta).any(1).astype(int)   # 1 if classified as in distribution

    ood_error = 1. - metrics.accuracy_score(ood_labels, all_pred)
    print('OOD Error:', ood_error)
    result_collection['OOD error'] = ood_error

    p = pred.max(1)
    ood_auc = metrics.roc_auc_score(ood_labels, p)
    result_collection['OOD AUC'] = ood_auc
    print('OOD Area under Curve:', ood_auc)

    # comparison of anomaly score and misclassification
    erroneous_prediction = y_true != pred_ds.argmax(1)
    ood_labels = np.array(ood_labels)
    anomaly_score = pred_ds.max(1)
    auc = 1. - metrics.roc_auc_score(erroneous_prediction, anomaly_score)
    result_collection['AUC anomaly score and misclassification'] = auc

    clf_error = 1. - metrics.accuracy_score(erroneous_prediction, anomaly_score < 0.5)
    result_collection['Classification error anomaly prediction and misclassification'] = clf_error

    recall_error = 1. - metrics.recall_score(erroneous_prediction, anomaly_score < 0.5)
    result_collection['Recall anomaly prediction and misclassification'] = recall_error

    precision_error = 1. - metrics.precision_score(erroneous_prediction, anomaly_score < 0.5)
    result_collection['Precision anomaly prediction and misclassification'] = precision_error

    f1_error = 1. - metrics.f1_score(erroneous_prediction, anomaly_score < 0.5)
    result_collection['F1 Score anomaly prediction and misclassification'] = f1_error

    fpr = fpr95(ood_labels, p)
    result_collection['FPR at 95% TPR'] = fpr
    print('FPR at 95% TPR:', fpr)

    return result_collection
