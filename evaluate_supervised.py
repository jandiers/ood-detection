from dataclasses import replace

from make_datasets import Dataset, val_split, Food101
from sklearn import metrics, ensemble
import numpy as np
import pandas as pd


def gradient_boosting_results(ds: Dataset, ood: Dataset) -> dict:
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

    # fit supervised classifier
    val_ds_ = replace(ds, split=val_split).load()
    val_ood_ = Food101(split=val_split).load()

    X = model.predict(val_ds_, verbose=1)
    y = [0] * X.shape[0]

    X_out = model.predict(val_ood_, verbose=1)
    y += [1] * X_out.shape[0]
    X = np.vstack((X, X_out))

    clf = ensemble.GradientBoostingClassifier(random_state=29)
    clf.fit(X, y)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=5)

    print(f'Gradient Boosting: {ds.__class__.__name__} vs. {ood.__class__.__name__}')
    print(f'Average CV Accuracy {ds.__class__.__name__} vs. Food101: {scores.mean().round(2)}. Full scores:', scores)

    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [1] * len(pred_ood) + [0] * len(pred_ds)

    r = pd.DataFrame({'pred': clf.predict(pred), 'scores': clf.predict_proba(pred)[:, 1]})

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
