from pineai.db import PreprocessingCollection, collection_by_name
from pineai.tasks.evaluation import EvaluationTask
from pineai.tasks.split import SplitTask
from pineai.tasks.sync import SyncTask
from pineai.tasks.transformation import TransformationTask
from pineai.tasks.vectorization import VectorizationTask
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier

from serce.tasks.optuna import OptunaTask
from serce.transformers.prepare import PrepareTransformer
from serce.transformers.radiomics_features import RadiomicsFeatureTransformer

TASKS = []

negative = list(range(1, 7))
positive = list(range(2955, 2961))
artifacts = list(range(4091, 4097))
names = ([f'{i:04}_negative' for i in negative] +
         [f'{i:04}_positive' for i in positive] +
         [f'{i:04}_artifacts' for i in artifacts])


def make_tasks():
    source = 'cine'
    radiomics_task = TransformationTask(
        cin_name='10x10',
        pipeline=[
            RadiomicsFeatureTransformer()
        ],
        # docs_filter={'name': {'$in': names}}
    )
    TASKS.append(radiomics_task)
    TASKS.append(SyncTask('RadiomicsFeature'))

    split_tasks = []
    split_task = SplitTask(cin_name=radiomics_task.cout_name, test_size=0.2, cv=5, y_key='label',
                           random_state=7, docs_filter={f'error_{name}': None for name in source.split('+')})
    split_tasks.append(split_task)
    TASKS.append(split_task)
    TASKS.append(SyncTask('Split'))

    # raw_task = TransformationTask(
    #     split_name=split_task.dout_name,
    #     pipeline=[
    #         PrepareTransformer(source=source)
    #     ]
    # )
    # TASKS.append(raw_task)
    # TASKS.append(SyncTask('Raw'))

    pca_task = TransformationTask(
        split_name=split_task.dout_name,
        pipeline=[
            PrepareTransformer(source=source, pca=5)
        ],
    )
    TASKS.append(pca_task)
    TASKS.append(SyncTask('PCA'))

    # kbest_task = TransformationTask(
    #     split_name=split_task.dout_name,
    #     pipeline=[
    #         PrepareTransformer(source=source, kbest_k=5, kbest_func=f_classif)
    #     ],
    # )
    # TASKS.append(kbest_task)
    # TASKS.append(SyncTask('KBest'))

    vect_tasks = []
    for task in [pca_task]:
        vect_task = VectorizationTask(cin_name=task.cout_name,
                                      split_name=split_task.dout_name,
                                      X_keys=['X'])
        vect_tasks.append(vect_task)
    TASKS.extend(vect_tasks)
    TASKS.append(SyncTask('Vect'))
    opt_tasks = []

    params = [([('KNN', KNeighborsClassifier())], {
        'KNN__weights': ['categorical', dict(choices=['uniform', 'distance'])],
        'KNN__p': ['float', dict(low=1, high=2)],
        'KNN__n_neighbors': ['int', dict(low=1, high=4)],
        # }), ([('nuSVC', NuSVC())], {
        #     'nuSVC__nu': [round(a, 2) for a in np.arange(0.01, 1.01, 0.01)],
        #     'nuSVC__kernel': ['linear', 'rbf', 'sigmoid'],
        #     'nuSVC__gamma': ['scale', 'auto'] + [round(a, 1) for a in np.arange(0.1, 1.01, 0.1)],
        # }), ([('MLP', MLPClassifier())], {
        #     'MLP__hidden_layer_sizes': [[size] * layers for size in [16, 32, 64, 128] for layers in [2, 3]],
        #     'MLP__activation': ['relu', 'tanh'],
    })]

    for vect_task in vect_tasks:
        for pipeline, param_grid in params:
            opt_task = OptunaTask(vect_name=vect_task.dout_name, pipeline=pipeline, params=param_grid, n_trials=10)
            opt_tasks.append(opt_task)

    TASKS.extend(opt_tasks)
    TASKS.append(SyncTask('Optuna'))
