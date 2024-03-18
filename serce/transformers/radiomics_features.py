import traceback

import SimpleITK as sitk
import numpy as np
from pineai.transformer.base import BaseTransformer
from radiomics import featureextractor


class RadiomicsFeatureTransformer(BaseTransformer):

    def __init__(self, no_mask=False):
        self.no_mask = no_mask

    @property
    def params(self):
        d = {}
        if self.no_mask:
            d['no_mask'] = True
        return d

    def transform_doc(self, doc):
        params = {
            'binWidth': 25,
            'normalize': True,
            'normalizeScale': 1,
            # 'resampledPixelSpacing': [1, 1],
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True,
            'enableParallel': False,
            # 'resegmentRange': None,
            'label': 1,
            'additionalInfo': True
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        doc['features'] = {}
        doc['diagnostics'] = {}
        if self.no_mask is None:
            mask = None
        else:
            mask = sitk.GetImageFromArray(doc['mask'])
        for source in ['cine', 'optical_flow', 'registration_transform']:

            try:
                img = doc[source]
                if source != 'cine':
                    img = np.sqrt(np.square(img[:, :, 0]) + np.square(img[:, :, 1]))
                image = sitk.GetImageFromArray(img)
                result = extractor.execute(image, mask)
                features = {}
                diagnostics = {}
                for k, v in result.items():
                    if k.startswith('diagnostics'):
                        diagnostics[k] = v
                    else:
                        features[k] = v

                doc['diagnostics'][source] = diagnostics
                doc['features'][source] = features
            except Exception as e:
                doc[f'error_{source}'] = str(e)
                doc[f'exception_{source}'] = traceback.format_exc()
