from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn import metrics

from datasets.make_datasets import Dataset


def baseline_results(ds: Dataset, ood: Dataset) -> dict:
    from conf import conf, normal
    result_collection = dict()

    # load model
    conf = replace(conf, strategy=normal, in_distribution_data=ds, out_of_distribution_data=None)
    model = conf.make_model()
    model.load_weights(conf.checkpoint_filepath)

    # true labels and predictions
    y_true = np.hstack([y.numpy() for (x, y, w) in ds.load()])
    pred_ds = model.predict(ds.load(), verbose=1)
    pred_ood = model.predict(ood.load(), verbose=1)

    threshold = np.percentile(pred_ds.max(1), 5)  # percentile as threshold for ood-classfication
    result_collection['threshold_ood'] = threshold

    print(f'Baseline: {ds.__class__.__name__} vs. {ood.__class__.__name__}')

    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

    r = pd.DataFrame({'pred': (pred > threshold).any(1).astype(int), 'scores': pred.max(1)})

    ood_error = 1. - metrics.accuracy_score(ood_labels, r.pred)
    print('OOD Error:', ood_error)
    result_collection['OOD error'] = ood_error

    scores = r.scores
    ood_auc = metrics.roc_auc_score(ood_labels, scores)
    result_collection['OOD AUC'] = ood_auc
    print('OOD Area under Curve:', ood_auc)

    # comparison of anomaly score and misclassification
    erroneous_prediction = y_true != pred_ds.argmax(1)
    ood_labels = np.array(ood_labels)
    clf_scores = r[ood_labels == 0]

    try:
        auc = metrics.roc_auc_score(erroneous_prediction, clf_scores['scores'].values)
    except ValueError:
        auc = -123456789

    result_collection['AUC anomaly score and misclassification'] = auc

    def fpr95(y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        ix = np.argwhere(tpr >= 0.95).ravel()[0]
        return fpr[ix]

    fpr = fpr95(ood_labels, scores)
    result_collection['FPR at 95% TPR'] = fpr
    print('FPR at 95% TPR:', fpr)

    return result_collection
