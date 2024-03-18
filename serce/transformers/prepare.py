import numpy as np
from pineai.db import Document

from pineai.transformer.base import BaseTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler


class PrepareTransformer(BaseTransformer):
    use_parallel = False

    def __init__(self, source, pca=None, kbest_k=None, kbest_func=None):
        self.source = source
        self.pca = pca
        self.kbest_k = kbest_k

        if pca:
            self.pca_reductor = PCA(n_components=self.pca)
        if kbest_k:
            self.feature_names = None
            self.kbest_func = kbest_func.__name__
            self.kbest_reductor = SelectKBest(kbest_func, k=kbest_k)

    @property
    def params(self):
        d = {
            'source': self.source
        }
        if self.pca:
            d['pca'] = self.pca
        if self.kbest_k:
            d['kbest_k'] = self.kbest_k
            d['kbest_func'] = self.kbest_func
        return d

    def get_X(self, doc: Document, source: str) -> np.ndarray:
        X = []
        for name in source.split('+'):
            for k, v in sorted(doc['features'][name].items()):
                X.append(v)
        return np.array(X)

    def fit(self, docs):
        X, y = [], []
        for doc in docs:
            X.append(self.get_X(doc, self.source))
            y.append(doc['label'])
        X = np.array(X)
        y = np.array(y)
        if self.pca:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.pca_reductor.fit(X_scaled)
        elif self.kbest_k:
            if self.feature_names is None:
                self.feature_names = []
                for name in self.source.split('+'):
                    for f in sorted(doc['features'][name]):
                        self.feature_names.append(f'{name}_{f}')
                self.feature_names = np.array(self.feature_names)
            self.kbest_reductor.fit(X, y, )

    def transform_doc(self, doc):
        doc['source'] = self.source
        X = self.get_X(doc, self.source)
        if self.pca:
            X = self.pca_reductor.transform([X])[0]
            doc['pca'] = self.pca
        if self.kbest_k:
            X = self.kbest_reductor.transform([X])[0]
            doc['kbest_k'] = self.kbest_k
            doc['kbest_func'] = self.kbest_func
            mask = self.kbest_reductor.get_support()
            doc['kbest_feature_names'] = self.feature_names[mask].tolist()
        doc['X'] = X
