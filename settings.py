import pathlib

BASE_DIR = pathlib.Path(__file__).parent.resolve()

DATA_DIR = (BASE_DIR / '..' / 'dane').resolve()
CACHE_FILE = (BASE_DIR / '..' / 'dane' / 'cache.pkl').resolve()

MAX_VALUE = 2**12
