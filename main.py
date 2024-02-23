import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import SimpleITK as sitk
import dask.distributed
import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from pycimg import CImg
from radiomics import featureextractor
from scipy.stats import randint, uniform
from simple_settings import settings
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.environ.setdefault('SIMPLE_SETTINGS', 'settings')

@dataclass
class Result:
    prep: str
    method: str
    value: float
    params: dict

@dataclass
class Sample:
    label: int  # 0 - serce jest zdrowe, 1 - w o brębie miokarbium są blizny
    cine: np.ndarray  # oryginał - rozkurczone
    cine_delayed: np.ndarray  # ten sam obszar co cine, ale w trakcie 1/4 skurczu
    optical_flow: np.ndarray  # optical flow - przepływ z opencv
    registration_transform: np.ndarray  # registration - inna transformacja
    mask: np.ndarray  # gdzie jest ściana serca, wyznaczana automatem, 1 to nasz obszar do szukania
    features: dict = None


def _extract_features(params, sample):
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    image = sitk.GetImageFromArray(sample.cine)
    mask = sitk.GetImageFromArray(sample.mask)
    result = extractor.execute(image, mask)

    # for feature_name in result.keys():
    #     print(f"{feature_name}: {result[feature_name]}")
    sample.features = {k: v for k, v in result.items() if not k.startswith('diagnostics')}

    # print(sample.features)
    return sample.features


class Main:
    values: list[Sample] = []

    def __init__(self):
        if settings.CACHE_FILE.exists():
            self.load_data()
        else:
            self.data = []
            for file in settings.DATA_DIR.iterdir():
                arr = np.load(file, allow_pickle=True)
                print(f'File "{file}": {arr.size} samples')
                for d in arr:
                    self.data.append(Sample(**d))
            self.save_data()

    def load_data(self):
        self.data = pickle.loads(settings.CACHE_FILE.read_bytes())
        print(f'File "{settings.CACHE_FILE}": {len(self.data)} samples')

    def save_data(self):
        settings.CACHE_FILE.write_bytes(pickle.dumps(self.data))

    def show_img(self, arr):
        img = CImg(arr)
        img.display()

    def stats(self):
        label = 0
        max_value = 0
        min_value = sys.maxsize
        print(f"Samples: {len(self.data)}")
        for d in self.data:
            label += d.label
            if max_value < d.cine.max():
                max_value = d.cine.max()
            if min_value > d.cine.min():
                min_value = d.cine.min()
        print(f'Min value: {min_value}')
        print(f'Max value: {max_value}')
        print(f'Label: {label}')

    def gather_features(self):
        params = {
            'binWidth': 25,
            'normalize': True,
            'normalizeScale': 1,
            # 'resampledPixelSpacing': [1, 1],
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True,
            'enableParallel': True,
            # 'resegmentRange': None,
            'label': 1,
            'additionalInfo': True
        }

        results = Parallel(n_jobs=-1)(delayed(_extract_features)(params, sample) for sample in self.data)
        for featues, sample in zip(results, self.data):
            sample.features = featues
        self.save_data()

    # Pearson correlation coefficient
    def pcc(self):
        df = pd.DataFrame({'label': s.label} | s.features for s in self.data)
        correlations = {}
        for method in ['pearson', 'kendall', 'spearman']:
            correlations[method] = df.corrwith(df['label'], method=method).abs().sort_values(ascending=False)
        correlations_df = pd.DataFrame(correlations)
        correlations_df.to_csv(settings.ANALYSIS_DIR / f'correlations.csv')
        return correlations_df

    @property
    def dataset(self):
        X = pd.DataFrame(s.features for s in self.data)
        y = np.array([s.label for s in self.data])
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
                X, y = self.dataset
                correlations = self.pcc()
                keys = correlations[kind].sort_values(ascending=False)[1:11].keys().to_list()
                X = X[keys]
            case 'pca':
                X, y = self.pca()
            case kind if kind.startswith('original_'):
                X, y = self.dataset
                X = X[[col for col in X.columns if col.startswith(kind)]]
            case _:
                raise Exception(f'Invalid preprocessing option: {kind}')
        toc = time.perf_counter()
        print(f'Preprocessing: {kind}, Time: {toc - tic:.1f}')
        return X, y

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        match method:
            case 'knn':
                def optimize(trial: optuna.Trial):
                    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
                    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                    p = trial.suggest_float('p', 1, 2)
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
                    return cross_val_score(knn, X_train, y_train, cv=5).mean()
            case 'rf':
                def optimize(trial: optuna.Trial):
                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                    max_depth = trial.suggest_int('max_depth', 5, 50)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    return cross_val_score(rf, X_train, y_train, cv=5).mean()
            case 'svm':
                def optimize(trial: optuna.Trial):
                    C = trial.suggest_float('C', 0.1, 10, log=True)
                    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                    svm = SVC(C=C, gamma=gamma, kernel=kernel)
                    return cross_val_score(svm, X_train, y_train, cv=5).mean()
            case _:
                raise Exception('Invalid classifier method')
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize, n_trials=100, n_jobs=-1)

        toc = time.perf_counter()
        result = Result(prep=prep, method=method,params=study.best_params, value=study.best_value)
        print(f'Method: {method}, Time: {toc - tic:.1f}, {result}')
        return result


def worker(prep, method):
    main = Main()
    # return Result(params={'a': 1}, value=1, prep=prep, method=method)
    return main.classify_optuna(prep, method)

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

if __name__ == '__main__':
    if len(sys.argv) == 3:
        prep = sys.argv[1]
        method = sys.argv[2]

        main = Main()
        main.stats()
        # main.gather_features()
        # main.classify(prep, method)
        result = main.classify_optuna(prep, method)

        print(f"Params: {result['params']}")
        print(f"Value: {result['value']}")

    else:
        # ustawianie niższego priorytetu obliczeń niż procesy w systemie
        os.nice(10)

        methods = ['knn', 'rf', 'svm']
        preps = ['raw', 'pca', 'pearson']
        results: list[Result] = joblib.Parallel(n_jobs=-1, backend='multiprocessing')(joblib.delayed(worker)(prep, method)
                                             for prep in preps for method in methods)
        values = defaultdict(dict)
        params = defaultdict(dict)
        for r in results:
            values[r.method][r.prep] = r.value
            params[r.method][r.prep] = r.params
        table(values)
        table(params)


