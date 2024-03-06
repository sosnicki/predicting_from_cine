import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from typing import Any, Optional

import SimpleITK as sitk
import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from pycimg import CImg
from radiomics import featureextractor
from scipy.stats import randint, uniform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import settings
from dc import Result, Feature, Sample


def _extract_features(source, params, sample):
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        img = getattr(sample, source)
        if source != 'cine':
            img = np.sqrt(np.square(img[:, :, 0]) + np.square(img[:, :, 1]))
        image = sitk.GetImageFromArray(img)
        mask = sitk.GetImageFromArray(sample.mask)
        result = extractor.execute(image, mask)
        features = {k.replace('-', '_'): v for k, v in result.items()}
        return Feature(source=source, **features)
    except Exception as e:
        logging.error(e)


class Main:
    data: list[Sample]
    values: list[Sample] = []

    def __init__(self, source: str, prefix: str, sampler: Optional[str], random_state: int):
        self.prefix = prefix
        self.source = source
        self.sampler = sampler
        self.random_state = random_state
        self.log = logging.getLogger('main')
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(logging.FileHandler(settings.ANALYSIS_DIR / 'main.log'))
        if self.sampler:
            self.sampling()
        else:
            self.gather_features()

    def load(self):
        cache_file = settings.DATA_DIR / f'cache{self.prefix}.pkl'
        if cache_file.exists():
            self.data = pickle.loads(cache_file.read_bytes())
            print(f'File "{cache_file}": {len(self.data)} samples')
        else:
            self.data = []
            for name in ['negative', 'positive', 'artifacts']:
                file = settings.DATA_DIR / f'{name}{self.prefix}.npy'
                if not file.name.endswith('.npy'):
                    continue
                arr = np.load(file, allow_pickle=True)
                print(f'File "{file}": {arr.size} samples')
                for d in arr:
                    self.data.append(Sample(**d))
            cache_file.write_bytes(pickle.dumps(self.data))

    def show_img(self, arr):
        img = CImg(arr)
        img.display()

    def parallel(self, function, param_tuples):
        if sys.gettrace():
            for params in param_tuples:
                function(*params)
        else:
            return Parallel(n_jobs=-1, backend='multiprocessing')(delayed(function)(*params)
                                                                  for params in param_tuples)

    def gather_features(self):
        features_file = settings.DATA_DIR / f'features{self.prefix}_{self.source}.pkl'
        if features_file.exists():
            self.data = pickle.loads(features_file.read_bytes())
        else:
            self.load()
            params = {
                'binWidth': 25,
                'normalize': True,
                'normalizeScale': 1,
                # 'resampledPixelSpacing': [1, 1],
                'interpolator': 'sitkBSpline',
                'enableCExtensions': True,
                'enableParallel': False,
                # 'resegmentRange': None,
                'label': 1,
                'additionalInfo': True
            }

            features = self.parallel(_extract_features, [(self.source, params, sample) for sample in self.data])
            filtered_data = []
            for d, f in zip(self.data, features):
                if f is None:
                    continue
                d.features = f
                filtered_data.append(d)
            self.data = filtered_data

            features_file.write_bytes(pickle.dumps(self.data))

    def sampling(self):
        sampling_file = settings.DATA_DIR / (f'sampling{self.prefix}_{self.source}_'
                                             f'{self.sampler}_{self.random_state}.pkl')
        if sampling_file.exists():
            self.data = pickle.loads(sampling_file.read_bytes())
        else:
            self.gather_features()

            X, y = self.dataset
            print(self.sampler)
            print('before', Counter(y))
            if self.sampler == 'OverSampler':
                ros = RandomOverSampler(random_state=self.random_state)
            elif self.sampler == 'UnderSampler':
                ros = RandomUnderSampler(random_state=self.random_state)
            else:
                raise Exception('Invalid Sampler')
            X_res, y_res = ros.fit_resample(X, y)
            print('after', Counter(y_res))
            self.data = [self.data[i] for i in ros.sample_indices_]
            sampling_file.write_bytes(pickle.dumps(self.data))

    # Pearson correlation coefficient
    def pcc(self, kind: str):
        X, y = self.dataset
        correlations = X.corrwith(pd.Series(y), method=kind).abs().sort_values(ascending=False)
        keys = correlations[1:11].keys().to_list()
        X = X[keys]
        correlations_df = pd.DataFrame(correlations)
        correlations_df.to_csv(settings.ANALYSIS_DIR / f'{kind}.csv')
        return X, y

    @property
    def dataset(self):
        X, y = [], []
        for sample in self.data:
            X.append(sample.features.values)
            y.append(sample.label)
        X = pd.DataFrame(X)
        y = np.array(y)
        return X, y

    def pca(self):
        X, y = self.dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        # explained_variance_ratio = pca.explained_variance_ratio_
        # print("Explained Variance Ratio:", explained_variance_ratio)

        return X_pca, y

    def preprocess(self, kind) -> tuple[np.ndarray, np.ndarray]:
        tic = time.perf_counter()
        match kind:
            case 'raw':
                X, y = self.dataset
            case 'pearson' | 'kendall' | 'spearman':
                X, y = self.pcc(kind)
            case 'pca':
                X, y = self.pca()
            case kind if kind.startswith('original_'):
                X, y = self.dataset
                X = X[[col for col in X.columns if col.startswith(kind)]]
            case _:
                raise Exception(f'Invalid preprocessing option: {kind}')
        toc = time.perf_counter()
        self.log.info(f'Preprocessing: {kind}, Time: {toc - tic:.1f}')
        return X, y

    # stary kod, nie aktualizowany
    def classify(self, prep: str, method: str):
        tic = time.perf_counter()
        X, y = self.preprocess(prep)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Method: {method}")
        match method:
            case 'knn':
                knn_param_grid = {'n_neighbors': randint(1, 50), 'weights': ['uniform', 'distance'], 'p': [1, 2]}
                classifier = RandomizedSearchCV(KNeighborsClassifier(), knn_param_grid, n_iter=5, cv=5, n_jobs=-1,
                                                random_state=42)
            case 'rf':
                rf_param_grid = {'n_estimators': randint(50, 200), 'max_depth': [None] + list(randint(5, 50).rvs(5)),
                                 'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 20)}
                classifier = RandomizedSearchCV(RandomForestClassifier(), rf_param_grid, n_iter=5, cv=5, n_jobs=-1,
                                                random_state=42)
            case 'svm':
                svm_param_grid = {'C': uniform(0.1, 10), 'gamma': ['scale', 'auto'],
                                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
                classifier = RandomizedSearchCV(SVC(), svm_param_grid, n_iter=5, cv=5, n_jobs=-1, random_state=42)
            case _:
                raise Exception('Invalid classifier method')
        classifier.fit(X_train, y_train)
        accuracy = classifier.best_score_
        print("Best Parameters:", classifier.best_params_)
        print("Best Accuracy:", accuracy)
        toc = time.perf_counter()
        print(f'Time: {toc - tic:.1f}')

    def classify_optuna(self, prep: str, method: str):
        tic = time.perf_counter()
        X, y = self.preprocess(prep)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        match method:
            case 'knn':
                classifier_cls = KNeighborsClassifier

                def optimize(trial: optuna.Trial):
                    params = dict(
                        n_neighbors=trial.suggest_int('n_neighbors', 1, 50),
                        weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                        p=trial.suggest_float('p', 1, 2)
                    )
                    knn = KNeighborsClassifier(**params)
                    return cross_val_score(knn, X_train, y_train, cv=5).mean()
            case 'rf':
                classifier_cls = RandomForestClassifier

                def optimize(trial: optuna.Trial):
                    params = dict(
                        n_estimators=trial.suggest_int('n_estimators', 50, 200),
                        max_depth=trial.suggest_int('max_depth', 5, 50),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20)
                    )
                    rf = RandomForestClassifier(**params)
                    return cross_val_score(rf, X_train, y_train, cv=5).mean()
            case 'svm':
                classifier_cls = SVC

                def optimize(trial: optuna.Trial):
                    params = dict(
                        C=trial.suggest_float('C', 0.1, 10, log=True),
                        gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                        kernel=trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                    )
                    svm = SVC(**params)
                    return cross_val_score(svm, X_train, y_train, cv=5).mean()
            case 'gbc':
                classifier_cls = GradientBoostingClassifier

                def optimize(trial: optuna.Trial):
                    params = dict(
                        n_estimators=trial.suggest_int('n_estimators', 50, 200),
                        learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                        subsample=trial.suggest_float('subsample', 0.5, 1.0),
                        max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2', 10, 20, 30])
                    )
                    gbc = GradientBoostingClassifier(**params)
                    return cross_val_score(gbc, X_train, y_train, cv=5).mean()
            case _:
                raise Exception('Invalid classifier method')
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize, n_trials=100, n_jobs=-1)

        # Train a model with the best parameters
        best_model = classifier_cls(**study.best_params)
        best_model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = best_model.predict(X_test)

        # Calculate accuracy or any other metric
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        toc = time.perf_counter()
        result = Result(
            prep=prep,
            method=method,
            params=study.best_params,
            accuracy=accuracy,
            roc_auc=roc_auc,
            y_pred=y_pred,
            y_true=y_test,
            random_state=self.random_state,
            source=self.source,
            sampler=self.sampler
        )
        self.log.info(f'Method: {method}, Time: {toc - tic:.1f}, {result.accuracy}')
        return result


def worker(prefix: str, features_source: str, sampler: str, prep: str, method: str, random_state: int,
           queue: mp.Queue) -> Result:
    main = Main(features_source, prefix, sampler, random_state)
    try:
        # return Result(params={'a': 1}, value=1, prep=prep, method=method)
        result = main.classify_optuna(prep, method)
        queue.put(result)
        return result
    except:
        main.log.exception(f'Worker exception {features_source=} {prep=} {method=} {random_state=}')


def result_writer(prefix: str, queue: mp.Queue, results: list[Result]) -> None:
    results_file = settings.ANALYSIS_DIR / f'result{prefix}.pkl'
    while result := queue.get():
        results.append(result)
        results_file.write_bytes(pickle.dumps(results))


def table(data: dict[str, dict[str, Any]]):
    headers = {' '}
    for columns in data.values():
        headers.update(columns.keys())
    print('\t'.join(sorted(headers)))

    for row in sorted(data):
        row_texts = [row]
        for col in sorted(data[row]):
            row_texts.append(f"\t{data[row][col]}")
        print('\t'.join(row_texts))


def main():
    if len(sys.argv) < 2:
        print('Brak arguentów')
        return

    if sys.argv[1] == 'test':
        main = Main('cine', '', None, 42)
        result = main.classify_optuna('pca', 'knn')
        print(f"Params: {result.params}")
        print(f"Accuracy: {result.accuracy}")
    elif sys.argv[1] == 'prepare':
        for source in settings.SOURCES:
            for prefix in settings.PREFIXES:
                for sampler in settings.SAMPLERS:
                    for rs in settings.RANDOM_STATES:
                        Main(source, prefix, sampler, rs)
    elif sys.argv[1] == 'classify':
        # ustawianie niższego priorytetu obliczeń niż procesy w systemie
        os.nice(10)
        manager = mp.Manager()

        for prefix in settings.PREFIXES:
            results = pickle.loads((settings.ANALYSIS_DIR / f'result{prefix}.pkl').read_bytes())
            shared_queue = manager.Queue()
            p = mp.Process(target=result_writer, args=(prefix, shared_queue, results))
            p.start()

            methods = ['knn', 'rf', 'gbc']
            preps = ['raw', 'pca', 'pearson']
            # preps = ['pca']
            results_set = {(prefix, r.source, r.sampler, r.prep, r.method, r.random_state) for r in results}
            params = []
            for prep in preps:
                for method in methods:
                    for source in settings.SOURCES:
                        for sampler in settings.SAMPLERS:
                            for rs in settings.RANDOM_STATES:
                                if (prefix, source, sampler, prep, method, rs) in results_set:
                                    print('Skip', (prefix, source, sampler, prep, method, rs))
                                else:
                                    params.append((prefix, source, sampler, prep, method, rs, shared_queue))
            joblib.Parallel(n_jobs=-1, backend='multiprocessing')(map(lambda x: joblib.delayed(worker)(*x), params))
            p.terminate()


if __name__ == '__main__':
    main()
