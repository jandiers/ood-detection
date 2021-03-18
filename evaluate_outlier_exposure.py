from dataclasses import replace
from conf import conf, outlier_exposure
from make_datasets import Cifar10, Cifar100, Textures, SVHNCropped, test_split, val_split
from sklearn import metrics
import numpy as np

# in and out of distribution data
ds = Cifar100(test_split)
ood = Textures()

# load model
conf = replace(conf, strategy=outlier_exposure, in_distribution_data=ds, out_of_distribution_data=None)
model = conf.make_model()
model.load_weights(conf.checkpoint_filepath)

# true labels and predictions
y_true = np.hstack([y.numpy() for (x, y, w) in ds.load()])
pred_ds = model.predict(ds.load(), verbose=1)
pred_ood = model.predict(ood.load(), verbose=1)

print(f'Outlier Exposure: {ds.__class__.__name__} vs. {ood.__class__.__name__}')

class_accuracy = metrics.accuracy_score(y_true, pred_ds.argmax(1))
print('Classification Accuracy no OOD:', class_accuracy)

# OOD accuracies
ood_detected = (pred_ood > 0.5).any(1).astype(int)   # 1 if classified as in distribution
in_dist_detected = (pred_ds > 0.5).any(1).astype(int)

# concat pred for in-dist and out-of-dist
all_pred = np.hstack((ood_detected, in_dist_detected))

# labels: 0 for in distribution, 1 for out of distribution
ood_labels = [0] * len(pred_ood) + [1] * len(pred_ds)

ood_accuracy = metrics.accuracy_score(ood_labels, all_pred)
print('OOD Accuracy:', ood_accuracy)

p = np.vstack((pred_ood, pred_ds))
p = p.max(1)
ood_auc = metrics.roc_auc_score(ood_labels, p)
print('OOD Area under Curve:', ood_auc)


def fpr95(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ix = np.argwhere(tpr >= 0.95).ravel()[0]
    return fpr[ix]


fpr = fpr95(ood_labels, p)
print('FPR at 95% TPR:', fpr)
