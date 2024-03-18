from pineai.tasks.sync import SyncTask
from pineai.tasks.transformation import TransformationTask
from pineai.tasks.vectorization import VectorizationTask
from sklearn.feature_selection import f_classif, mutual_info_classif

from serce.transformers.prepare import PrepareTransformer
from .e1_features import TASKS, split_task_dict, make_tasks as features_make_tasks

vect_tasks = []


def make_tasks():
    features_make_tasks()
    prepare_tasks = []
    for source, split_tasks in split_task_dict.items():
        for split_task in split_tasks:
            raw_task = TransformationTask(
                split_name=split_task.dout_name,
                pipeline=[
                    PrepareTransformer(source=source)
                ]
            )
            prepare_tasks.append((split_task, raw_task))

            for n in [5, 10, 20]:
                pca_task = TransformationTask(
                    split_name=split_task.dout_name,
                    pipeline=[
                        PrepareTransformer(source=source, pca=n)
                    ],
                )
                prepare_tasks.append((split_task, pca_task))

            for n in [10, 20]:
                for func in [f_classif, mutual_info_classif]:
                    kbest_task = TransformationTask(
                        split_name=split_task.dout_name,
                        pipeline=[
                            PrepareTransformer(source=source, kbest_k=n, kbest_func=func)
                        ],
                    )
                    prepare_tasks.append((split_task, kbest_task))

    TASKS.extend(i[1] for i in prepare_tasks)
    TASKS.append(SyncTask(f'Prepare'))
    for split_task, prepare_task in prepare_tasks:
        vect_task = VectorizationTask(cin_name=prepare_task.cout_name,
                                      split_name=split_task.dout_name,
                                      X_keys=['X'])
        vect_tasks.append(vect_task)

    TASKS.extend(vect_tasks)
    TASKS.append(SyncTask(f'Vect'))
