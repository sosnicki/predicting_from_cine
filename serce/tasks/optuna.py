import logging
import traceback
import warnings
from collections import defaultdict

import numpy as np
import optuna
from bson import DBRef
from django.conf import settings
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pineai.computation import escape_Function_in_dict, escape_Function_in_list
from pineai.db import collection_by_name
from pineai.tasks.base import BaseTask
from pineai.tasks.vectorization import VectorizationTask

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

_logger = logging.getLogger('pineai.tasks.optimization')


class OptunaTask(BaseTask):
    """
    Task which performs optimalization on document `din` of `cin` collection. Results are stored in `cout` collection.

    Args:
        vect_name (str): input document name.
        cout_name (str): output collection name.
        pipeline (list[tuple]): List of tuples (name, estimator) for constructing Scalers and Models.
        params (dics): parameters for estimators as in ``GridSearchCV``.
        cv (int): input collection name, default is `data`.
        cin_name (str): input collection name, default is `data`.
        n_jobs (int): as `n_jobs` in https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html, default -1.
    """
    cout_name = 'opt'
    cin_name = VectorizationTask.cout_name
    dout_name = None

    def __init__(self, *, vect_name, pipeline, params, n_trials, n_jobs=settings.N_JOBS):
        self.vect_name = vect_name
        self.pipeline = pipeline
        self.params = params
        self.n_trials = n_trials
        self.n_jobs = n_jobs

        param_names = defaultdict(list)

        for k, v in sorted(self.params.items()):
            name, param = k.split('__', 1)
            param_names[name].append('{}={}'.format(param, v))

        self.names = []
        self.methods = []
        for name, estimator in self.pipeline:
            self.names.append('{}({})'.format(name, ', '.join(param_names[name])))
            self.methods.append(name)

        self.dout_name = '{} [{}] {}'.format(self.vect_name, '->'.join(self.names), n_trials)
        self.cout = collection_by_name(self.cout_name)

    @property
    def name(self) -> str:
        return f'Optimization {self.dout_name}'

    @property
    def expected_count(self) -> int:
        return 1

    @property
    def created_count(self) -> int:
        return self.cout.count({'name': self.dout_name})

    @property
    def failed_count(self) -> int:
        return self.cout.count({'name': self.dout_name, 'error': {'$exists': True}})

    @property
    def completed(self):
        return self.cout.count({'name': self.dout_name}) > 0

    def run(self):
        cin = collection_by_name(self.cin_name)
        din = cin.find_doc({'name': self.vect_name})
        count = self.cout.count({'name': self.dout_name})
        if count > 0:
            _logger.warning(f'Document {self.dout_name} already exists in {self.cout_name} collection. Skipping.')
            return

        folds = []
        for i in range(din['split']['cv']):
            folds.append((np.argwhere(din['X_train_splits'] == i).flatten(),
                          np.argwhere(din['X_val_splits'] == i).flatten() + din['X_train'].shape[0]))
        X_trainval = np.vstack((din['X_train'], din['X_val']))
        y_trainval = np.hstack((din['y_train'], din['y_val']))
        X_test = din['X_test']
        y_test = din['y_test']

        dout = self.cout.new_doc({
            'name': self.dout_name,
            'cin': self.cin_name,
            'cout': self.cout_name,
            'din_name': din['name'],
            'din': DBRef(self.cin_name, din['_id']),
            'n_trials': self.n_trials,
            'methods': self.methods,
            'params': escape_Function_in_dict(self.params)
        })
        try:

            classifier_cls = Pipeline(self.pipeline)

            def optimize(trial: optuna.Trial):
                params = {}
                for name, (kind, kwargs) in self.params.items():
                    kwargs.setdefault('name', name)
                    params[name] = getattr(trial, f'suggest_{kind}')(**kwargs)
                knn = classifier_cls.set_params(**params)
                return cross_val_score(knn, X_trainval, y_trainval, cv=folds).mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(optimize, n_trials=self.n_trials, n_jobs=-1)

            # Train a model with the best parameters
            best_model = classifier_cls.set_params(**study.best_params)
            best_model.fit(X_trainval, y_trainval)

            # Make predictions on the validation set
            y_pred = best_model.predict(X_test)

            # Calculate accuracy or any other metric
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate ROC AUC score
            roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            dout.update({
                'best_params': study.best_params,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_true': y_test,
            })
        except Exception as e:
            dout['error'] = f'Optimalization failed: {e}'
            dout['exception'] = traceback.format_exc()

        dout.save()
