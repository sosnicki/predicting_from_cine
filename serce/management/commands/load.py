from importlib import import_module
from multiprocessing import Process, Queue

from django.conf import settings
from django.core.management import BaseCommand
from django.utils.functional import empty
from pineai.db import MongoClient, collection_by_name
import numpy as np

class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('collection')
        parser.add_argument('--prefix', default='')


    def handle(self, *args, **options):

        coll = collection_by_name(options['collection'])
        nr = 1
        prefix = options['prefix']
        for name in ['negative', 'positive', 'artifacts']:
            print(name)
            file = settings.SOURCE_DATA_DIR / f'{name}{prefix}.npy'
            arr = np.load(file, allow_pickle=True)
            print(f'File "{file}": {arr.size} samples')
            for d in arr:
                # print(d)
                doc = coll.new_doc(d)
                doc['nr'] = nr
                doc['kind'] = name
                doc['name'] = f'{nr:04}_{name}'
                doc['diff'] = d['cine'] - d['cine_delayed']
                nr += 1
                doc.save()
