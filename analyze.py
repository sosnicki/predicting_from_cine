import pickle
from dataclasses import asdict

import pandas as pd

import settings
from dc import Result


class Analyzer:
    results: list[Result]

    def __init__(self):
        results_file = settings.ANALYSIS_DIR / 'result.pkl'
        self.results = pickle.loads(results_file.read_bytes())

    def run(self):
        df = pd.DataFrame([asdict(r) for r in self.results])
        df['samples'] = df['y_pred'].apply(lambda x: len(x))
        print(df)
        keys = [
            'samples',
            'prep',
            'method',
            'params',
            'accuracy',
            'roc_auc',
            'random_state',
        ]
        df[keys].to_csv(settings.ANALYSIS_DIR / 'result.csv')


if __name__ == '__main__':
    Analyzer().run()
