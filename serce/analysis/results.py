from collections import Counter

import pandas as pd
from pineai.db import collection_by_name, PreprocessingCollection
from pineai.helpers.mp import parallel, get_pool, close_pool


def get_info(doc_id):
    opt_doc = collection_by_name('opt').find_doc(doc_id)
    vect_doc = opt_doc['din']
    split_doc = vect_doc['split']
    split_info = PreprocessingCollection.meta.by_id(split_doc['cin'])
    vect_info = PreprocessingCollection.meta.by_id(vect_doc['cin'])
    d = vect_info['pipeline'][0]['params']
    if 'pca' in d:
        d['prepare'] = 'pca'
    elif 'kbest' in d:
        d['prepare'] = 'kbest'
    else:
        d['prepare'] = 'raw'
    d['mask'] = bool(split_info['pipeline'][0]['params'].get('no_mask')) is False
    d.update(dict(
        set_name=split_info['parent'],
        test_size=len(split_doc['test_docs']),
        trainval_size=len(split_doc['trainval_docs']),
        y_true=str(Counter(opt_doc['y_true']))[8:-1],
        y_pred=str(Counter(opt_doc['y_pred']))[8:-1],
        method=opt_doc['methods'][0],
        params=opt_doc['best_params'],
        accuracy=opt_doc['accuracy'],
        roc_auc=opt_doc['roc_auc'],
        n_trials=opt_doc['n_trials'],
        random_state=split_doc['random_state'],
    ))
    return d


def main():
    get_pool()
    opt = collection_by_name('opt')
    data = parallel(get_info, [(d,) for d in opt.find({}, {'_id': 1})])
    df = pd.DataFrame(data)
    df.to_csv(f'results.csv')
    close_pool()
