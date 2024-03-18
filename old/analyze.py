import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd

import settings
from dc import Result, Sample


class Analyzer:
    def results(self):
        data = []
        for prefix in settings.PREFIXES:
            results_file = settings.ANALYSIS_DIR / f'result{prefix}.pkl'
            results = pickle.loads(results_file.read_bytes())
            data.extend([asdict(r) | {'set': prefix} for r in results])
        df = pd.DataFrame(data)
        df['samples'] = df['y_pred'].apply(lambda x: len(x))
        df['y_true'] = df['y_true'].apply(lambda x: str(Counter(x))[8:-1])
        df['y_pred'] = df['y_pred'].apply(lambda x: str(Counter(x))[8:-1])
        print(df)
        keys = [
            'set',
            'samples',
            'y_true',
            'y_pred',
            'source',
            'prep',
            'method',
            'params',
            'accuracy',
            'roc_auc',
            'random_state',
            'sampler'
        ]
        df[keys].to_csv(settings.ANALYSIS_DIR / f'results.csv')

    def count_labels(self, data):
        labels = [s.label for s in data]
        counter = Counter(labels)
        return f'{counter[0]}\t{counter[1]}'

    def stats(self):
        for prefix in settings.PREFIXES:
            max_value = 0
            min_value = sys.maxsize
            data = pickle.loads((settings.DATA_DIR / f'cache{prefix}.pkl').read_bytes())
            print(f'Samples "{prefix}":\t{len(data)}\t{self.count_labels(data)}')
            for d in data:
                if max_value < d.cine.max():
                    max_value = d.cine.max()
                if min_value > d.cine.min():
                    min_value = d.cine.min()
            print(f'Min value: {min_value}, Max value: {max_value}')
            for source in settings.SOURCES:
                data = pickle.loads((settings.DATA_DIR / f'features{prefix}_{source}.pkl').read_bytes())
                print(f'Features "{source}":\t{len(data)}\t{self.count_labels(data)}')
                for sampler in settings.SAMPLERS:
                    if not sampler:
                        continue
                    data = pickle.loads((settings.DATA_DIR / f'sampling{prefix}_{source}_{sampler}_176.pkl')
                                        .read_bytes())
                    print(f'Features "{source}" "{sampler}":\t{len(data)}\t{self.count_labels(data)}')


def main():
    analyzer = Analyzer()
    if sys.argv[1] == 'stats':
        analyzer.stats()
    elif sys.argv[1] == 'results':
        analyzer.results()


if __name__ == '__main__':
    main()
