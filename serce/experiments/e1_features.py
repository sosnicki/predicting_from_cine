from collections import defaultdict

import numpy as np
from pineai.tasks.split import SplitTask
from pineai.tasks.sync import SyncTask
from pineai.tasks.transformation import TransformationTask

from serce.transformers.radiomics_features import RadiomicsFeatureTransformer

TASKS = []

split_task_dict = defaultdict(list)

np.random.seed(7)
seeds = np.random.randint(low=1, high=1000, size=10)


def make_tasks():
    radiomics_tasks = []
    for cin_name in ['10x10', '30x30']:
        radiomics_task = TransformationTask(
            cin_name=cin_name,
            pipeline=[
                RadiomicsFeatureTransformer()
            ]
        )
        radiomics_tasks.append(radiomics_task)
        if cin_name == '10x10':
            no_mask_radiomics_task = TransformationTask(
                cin_name=cin_name,
                pipeline=[
                    RadiomicsFeatureTransformer(no_mask=True)
                ]
            )
            radiomics_tasks.append(no_mask_radiomics_task)
    TASKS.extend(radiomics_tasks)
    TASKS.append(SyncTask('RadiomicsFeature'))

    for source in ['cine', 'optical_flow', 'registration_transform', 'cine+optical_flow', 'cine+registration_transform']:
        for radiomics_task in radiomics_tasks:
            for seed in seeds:
                split_task = SplitTask(cin_name=radiomics_task.cout_name, test_size=0.2, cv=5, y_key='label',
                                       random_state=seed,
                                       docs_filter={f'error_{name}': None for name in source.split('+')})
                split_task_dict[source].append(split_task)

        TASKS.extend(split_task_dict[source])
        TASKS.append(SyncTask(f'Split {source}'))
