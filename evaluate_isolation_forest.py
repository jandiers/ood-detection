from dataclasses import replace

from sklearn.exceptions import NotFittedError

from make_datasets import Dataset, val_split
from sklearn import metrics, ensemble
import numpy as np
import pandas as pd


def isolation_forest_results(ds: Dataset, ood: Dataset, iforest_per_class: bool) -> bool:
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

    # fit isolation forest
    val_ds_ = replace(ds, split=val_split)
    X = model.predict(val_ds_.load(), verbose=1)

    if iforest_per_class:
        rfs = {i: ensemble.IsolationForest() for i in range(ds.NUM_CLASSES)}
        prd = X.argmax(1)
        for t in range(ds.NUM_CLASSES):
            subX = X[prd == t]
            if subX.shape[0] == 0:
                continue
            rfs[t].fit(subX)
    else:
        rf = ensemble.IsolationForest(random_state=29, n_jobs=10)
        rf.fit(X)

    print(f'{str(len(rfs)) if iforest_per_class else str(1)} '
          f'Isolation Forest(s): {ds.__class__.__name__} vs. {ood.__class__.__name__}')

    class_error = 1. - metrics.accuracy_score(y_true, pred_ds.argmax(1))
    result_collection['classification error'] = class_error
    print('Classification Error on dataset:', class_error)

    pred = np.vstack((pred_ood, pred_ds))
    ood_labels = [-1] * len(pred_ood) + [1] * len(pred_ds)

    def grouped_prediction(df):
        target = df.values.argmax(1)[0]
        try:
            p = rfs[target].predict(df)
            s = rfs[target].score_samples(df)
        except NotFittedError:
            print('classifier for class', target, 'not fitted. return -1 for outlier and scores.')
            p = -1
            s = -1
        df['pred'], df['scores'] = p, s
        return df

    if iforest_per_class:
        r = pd.DataFrame(pred).groupby(pred.argmax(1)).apply(grouped_prediction)
    else:
        r = pd.DataFrame({'pred': rf.predict(pred), 'scores': rf.score_samples(pred)})

    ood_error = 1. - metrics.accuracy_score(ood_labels, r.pred)
    print('OOD Error:', ood_error)
    result_collection['OOD error'] = ood_error

    scores = r.scores
    ood_auc = metrics.roc_auc_score(ood_labels, scores)
    result_collection['OOD AUC'] = ood_auc
    print('OOD Area under Curve:', ood_auc)

    def fpr95(y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        ix = np.argwhere(tpr >= 0.95).ravel()[0]
        return fpr[ix]

    fpr = fpr95(ood_labels, scores)
    result_collection['FPR at 95% TPR'] = fpr
    print('FPR at 95% TPR:', fpr)
    return result_collection
