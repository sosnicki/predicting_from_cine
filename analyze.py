import pickle
import sys
from dataclasses import asdict

import pandas as pd

import settings
from dc import Result


class Analyzer:
    results: list[Result]

    def __init__(self, prefix: str):
        self.prefix = prefix

    def run(self):
        results_file = settings.ANALYSIS_DIR / f'result{self.prefix}.pkl'
        self.results = pickle.loads(results_file.read_bytes())
        df = pd.DataFrame([asdict(r) for r in self.results])
        df['samples'] = df['y_pred'].apply(lambda x: len(x))
        print(df)
        keys = [
            'samples',
            'source',
            'prep',
            'method',
            'params',
            'accuracy',
            'roc_auc',
            'random_state',
        ]
        df[keys].to_csv(settings.ANALYSIS_DIR / f'result{self.prefix}.csv')

    def stats(self):
        ones = 0
        zeros = 0
        max_value = 0
        min_value = sys.maxsize
        data = pickle.loads((settings.DATA_DIR / f'cache{self.prefix}.pkl').read_bytes())
        print(f'Samples "{self.prefix}": {len(data)}')
        for d in data:
            if d.label == 1:
                ones += 1
            else:
                zeros += 1
            if max_value < d.cine.max():
                max_value = d.cine.max()
            if min_value > d.cine.min():
                min_value = d.cine.min()
        print(f'Min value: {min_value}, Max value: {max_value}')
        print(f'Ones: {ones}, Zeros: {zeros}')
        for source in settings.SOURCES:
            filtered_data, features = pickle.loads(
                (settings.DATA_DIR / f'features{self.prefix}_{source}.pkl').read_bytes())
            print(f'Features "{source}": {len(features)}')


def main():
    for prefix in settings.PREFIXES:
        analyzer = Analyzer(prefix)
        # analyzer.stats()
        analyzer.run()


if __name__ == '__main__':
    main()
