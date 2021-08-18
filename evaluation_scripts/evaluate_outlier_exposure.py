from dataclasses import replace

import numpy as np
from sklearn import metrics

from datasets.make_datasets import Dataset


def outlier_exposure_results(ds: Dataset, ood: Dataset) -> dict:
    from conf import conf, outlier_exposure
    result_collection = dict()

    # load model
    conf = replace(conf, strategy=outlier_exposure, in_distribution_data=ds, out_of_distribution_data=None)
    model = conf.make_model()
    model.load_weights(conf.checkpoint_filepath)

    # true labels and predictions
    y_true = np.hstack([y.numpy() for (x, y, w) in ds.load()])
    pred_ds = model.predict(ds.load(), verbose=1)
    pred_ood = model.predict(ood.load(), verbose=1)

    print(f'Outlier Exposure: {ds.__class__.__name__} vs. {ood.__class__.__name__}')

    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    # OOD accuracies
    threshold = np.percentile(pred_ds.max(1), 5)  # percentile as threshold for ood-classfication
    result_collection['threshold_ood'] = threshold
    ood_detected = (pred_ood > threshold).any(1).astype(int)  # 1 if classified as in distribution
    in_dist_detected = (pred_ds > threshold).any(1).astype(int)

    # concat pred for in-dist and out-of-dist
    all_pred = np.hstack((ood_detected, in_dist_detected))

    # labels: 0 for in distribution, 1 for out of distribution
    ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

    ood_error = 1. - metrics.accuracy_score(ood_labels, all_pred)
    print('OOD Error:', ood_error)
    result_collection['OOD error'] = ood_error

    p = np.vstack((pred_ood, pred_ds))
    p = p.max(1)
    ood_auc = metrics.roc_auc_score(ood_labels, p)
    result_collection['OOD AUC'] = ood_auc
    print('OOD Area under Curve:', ood_auc)

    # comparison of anomaly score and misclassification
    erroneous_prediction = y_true != pred_ds.argmax(1)
    ood_labels = np.array(ood_labels)
    anomaly_score = pred_ds.max(1)
    try:
        auc = metrics.roc_auc_score(erroneous_prediction, anomaly_score)
    except ValueError:
        auc = -123456789
    result_collection['AUC anomaly score and misclassification'] = auc

    def fpr95(y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        ix = np.argwhere(tpr >= 0.95).ravel()[0]
        return fpr[ix]

    fpr = fpr95(ood_labels, p)
    result_collection['FPR at 95% TPR'] = fpr
    print('FPR at 95% TPR:', fpr)

    return result_collection
