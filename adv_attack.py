from util import save_import_tensorflow
import foolbox as fb
import tqdm

tf = save_import_tensorflow('1')


def make_adv_attack(model, ds, ood, ood_prediction, epsilon=0.003, attack=fb.attacks.LinfPGD()):
    fmodel = fb.TensorFlowModel(model, bounds=(0, 255))
    fmodel = fmodel.transform_bounds((0, 1))

    ds_full = ds.concatenate(ood).unbatch()
    ood_pred_ds = tf.data.Dataset.from_tensor_slices(ood_prediction)

    ds_full = tf.data.Dataset.zip((ds_full, ood_pred_ds))
    ds_full = ds_full.batch(16)

    attacked_imgs = []
    for (x, y, _), ood_p in tqdm.tqdm(ds_full):
        raw, clipped, is_adv = attack(fmodel, x, y, epsilons=epsilon)  # 0.003 is about 1/255.
        raw, clipped, is_adv = raw.numpy(), clipped.numpy(), is_adv.numpy()
        attacked_imgs.append(clipped)

    return attacked_imgs
