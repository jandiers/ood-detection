from conf import conf
from evaluate_isolation_forest import isolation_forest_results
from evaluate_odin import odin_results
from evaluate_outlier_exposure import outlier_exposure_results
from make_datasets import Cifar10, Cifar100, Textures, SVHNCropped, CatsVsDogs, Cassava, Cars196, test_split
import pandas as pd
from pathlib import Path

# in and out of distribution data
datasets = [
    Cars196(test_split),
    Cassava(test_split),
    CatsVsDogs(test_split),
    Cifar10(test_split),
    Cifar100(test_split)
]

for ds in datasets:
    ds_name = ds.__class__.__name__
    results = dict()

    for ood in datasets + [Textures(), SVHNCropped()]:   # additional datasets for evaluation only
        ood_name = ood.__class__.__name__
        if ds_name == ood_name:
            continue

        results[(ood_name, 'ODIN')] = odin_results(ds, ood)
        results[(ood_name, 'outlier exposure')] = outlier_exposure_results(ds, ood)
        results[(ood_name, 'single isolation forest')] = isolation_forest_results(ds, ood, iforest_per_class=False)
        results[(ood_name, 'multiple isolation forests')] = isolation_forest_results(ds, ood, iforest_per_class=True)

    df = pd.DataFrame(results).round(4)

    p = (Path.cwd() / 'results')
    p.mkdir(exist_ok=True)

    df.to_excel(p / f'{ds_name}.xlsx')
    df.to_latex(p / f'{ds_name}.tex')


from pathlib import Path
import pandas as pd

fs = list(Path.cwd().glob('./results/*.xlsx'))

dfs = [pd.read_excel(f, engine='openpyxl', index_col=0, header=[0, 1]) for f in fs]

r = []
for d in fs:
    df = pd.read_excel(d, engine='openpyxl', index_col=0, header=[0, 1]).T
    df = df.mean(level=1)
    df['in distribution data'] = d.stem

    df = df.set_index('in distribution data', append=True)
    df = df.reorder_levels([1, 0])

    r.append(df)

r = pd.concat(r)
cols = [('average over 6 datasets', c) for c in r.columns]
r.columns = pd.MultiIndex.from_tuples(cols)

clf_error = r.pop(('mean over 6 datasets', 'classification error')).to_frame()
clf_error.index.names = ['dataset', 'method']
clf_error.columns = ['classification error']
caption = 'Classification error on datasets for various methods.'
clf_error.to_latex('./results/grouped_results/clfResult.tex', label='aggCLFResult', caption=caption,
                   float_format='%.2f')


r.to_excel('./results/grouped_results/grouped_results.xlsx')

caption = 'Results for out of distribution detection based on 6 different datasets. Values represent averages. ' \
          'While ODIN and Outlier Exposure require explicit optimization for out of distribution detection, our ' \
          'approach only requires a validation set to learn the outlier distribution. It works with any existing ' \
          'classifier without modification.'
r.to_latex('./results/grouped_results/aggOODResult.tex', label='aggOODResult', caption=caption, float_format='%.2f')


