from dataclasses import replace
from conf import conf, normal
from make_datasets import Cifar10, Cifar100, Textures, SVHNCropped, test_split, val_split
from sklearn import metrics, ensemble
import numpy as np
import pandas as pd

# in and out of distribution data
ds = Cifar100(test_split)
ood = Textures()

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

rfs = {i: ensemble.IsolationForest() for i in range(ds.NUM_CLASSES)}
prd = X.argmax(1)
for t in range(ds.NUM_CLASSES):
    subX = X[prd == t]
    rfs[t].fit(subX)

# rf = ensemble.IsolationForest(random_state=29, n_jobs=10)
# rf.fit(X)

print(f'Isolation Forest: {ds.__class__.__name__} vs. {ood.__class__.__name__}')

class_accuracy = metrics.accuracy_score(y_true, pred_ds.argmax(1))
print('Classification Accuracy no OOD:', class_accuracy)

pred = np.vstack((pred_ood, pred_ds))
ood_labels = [-1] * len(pred_ood) + [1] * len(pred_ds)


def grouped_prediction(df):
    target = df.values.argmax(1)[0]
    p = rfs[target].predict(df)
    s = rfs[target].score_samples(df)
    df['pred'], df['scores'] = p, s
    return df


r = pd.DataFrame(pred).groupby(pred.argmax(1)).apply(grouped_prediction)


ood_accuracy = metrics.accuracy_score(ood_labels, r.pred)
print('OOD Accuracy:', ood_accuracy)

scores = r.scores
ood_auc = metrics.roc_auc_score(ood_labels, scores)
print('OOD Area under Curve:', ood_auc)


def fpr95(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ix = np.argwhere(tpr >= 0.95).ravel()[0]
    return fpr[ix]


fpr = fpr95(ood_labels, scores)
print('FPR at 95% TPR:', fpr)



