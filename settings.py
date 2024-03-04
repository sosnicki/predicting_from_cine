import pathlib

BASE_DIR = pathlib.Path(__file__).parent.resolve()

DATA_DIR = (BASE_DIR / '..' / 'dane').resolve()
ANALYSIS_DIR = (BASE_DIR / '..' / 'analiza').resolve()
CACHE_FILE = (BASE_DIR / '..' / 'dane' / 'cache.pkl').resolve()

MAX_VALUE = 2**12
SOURCES = ['cine', 'registration_transform', 'optical_flow']
PREFIXES = ['', 'Small']
