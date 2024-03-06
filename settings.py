import pathlib

BASE_DIR = pathlib.Path(__file__).parent.resolve()

DATA_DIR = pathlib.Path('/mnt/dane/serce/dane')
ANALYSIS_DIR = (BASE_DIR / '..' / 'analiza').resolve()
CACHE_FILE = (BASE_DIR / '..' / 'dane' / 'cache.pkl').resolve()

MAX_VALUE = 2**12
SOURCES = ['cine', 'registration_transform', 'optical_flow']
PREFIXES = ['', 'Small']
SAMPLERS = [None, 'OverSampler', 'UnderSampler']

import numpy as np
np.random.seed(7)
RANDOM_STATES = np.random.randint(low=1, high=1000, size=10)
