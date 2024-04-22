from pineai.db import collection_by_name

src = collection_by_name('6623c9e892380c23127cae9c')
dst = collection_by_name('65f80bd4133211d0b6d7df9b')

for s, d in zip(src.find_docs({}, sort=[('name', 1)]), dst.find_docs({}, sort=[('name', 1)])):
    print(s['name'], d['name'])
    if 'diff' in s['diagnostics']:
        d['diagnostics']['diff'] = s['diagnostics']['diff']
    if 'diff' in s['features']:
        d['features']['diff'] = s['features']['diff']
    if 'error_diff' in s:
        d['error_diff'] = s['error_diff']
    if 'exception_diff' in s:
        d['exception_diff'] = s['exception_diff']
    d.save()