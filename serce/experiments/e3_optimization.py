from pineai.tasks.sync import SyncTask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from serce.tasks.optuna import OptunaTask
from .e2_prepare import TASKS, vect_tasks, make_tasks as prepare_make_tasks

opt_tasks = []


def make_tasks():
    prepare_make_tasks()
    params = [([('KNN', KNeighborsClassifier())], {
        'KNN__weights': ['categorical', dict(choices=['uniform', 'distance'])],
        'KNN__p': ['float', dict(low=1, high=2)],
        'KNN__n_neighbors': ['int', dict(low=1, high=50)],
    }), ([('RFB', RandomForestClassifier(class_weight='balanced'))], {
        'RFB__n_estimators': ['int', dict(low=50, high=200)],
        'RFB__max_depth': ['int', dict(low=5, high=50)],
        'RFB__min_samples_split': ['int', dict(low=2, high=20)],
        'RFB__min_samples_leaf': ['int', dict(low=1, high=20)],
    }), ([('GBC', GradientBoostingClassifier())], {
        'GBC__n_estimators': ['int', dict(low=50, high=200)],
        'GBC__learning_rate': ['float', dict(low=0.001, high=0.1, log=True)],
        'GBC__max_depth': ['int', dict(low=3, high=10)],
        'GBC__min_samples_split': ['int', dict(low=2, high=20)],
        'GBC__min_samples_leaf': ['int', dict(low=1, high=20)],
        'GBC__subsample': ['float', dict(low=0.5, high=1.0)],
        'GBC__max_features': ['categorical', dict(choices=[None, 'sqrt', 'log2', 10, 20, 30])]
    })]

    for vect_task in vect_tasks:
        for pipeline, param_grid in params:
            if pipeline[0][0] == 'GBC':
                n_trials_list = [100, 300]
            else:
                n_trials_list = [100]
            for n_trials in n_trials_list:
                opt_task = OptunaTask(vect_name=vect_task.dout_name, pipeline=pipeline, params=param_grid,
                                      n_trials=n_trials)
                opt_tasks.append(opt_task)

    TASKS.extend(opt_tasks)
    TASKS.append(SyncTask('Optuna'))
