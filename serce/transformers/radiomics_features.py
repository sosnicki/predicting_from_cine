import traceback

import SimpleITK as sitk
import numpy as np
from pineai.transformer.base import BaseTransformer
from radiomics import featureextractor


class RadiomicsFeatureTransformer(BaseTransformer):

    def __init__(self, no_mask=False):
        self.no_mask = no_mask
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            binWidth=25,
            normalize=True,
            normalizeScale=1,
            interpolator='sitkBSpline',
            enableCExtensions=True,
            enableParallel=False,
            label=1,
            additionalInfo=True
        )

    @property
    def params(self):
        d = {}
        if self.no_mask:
            d['no_mask'] = True
        return d

    def get_features(self, image: sitk.Image, mask: sitk.Image) -> tuple[dict, dict]:
        result = self.extractor.execute(image, mask)
        features = {}
        diagnostics = {}
        for k, v in result.items():
            if k.startswith('diagnostics'):
                diagnostics[k] = v
            else:
                features[k] = v

        return diagnostics, features

    def transform_doc(self, doc):

        doc['features'] = {}
        doc['diagnostics'] = {}
        if self.no_mask is None:
            mask = None
        else:
            mask = sitk.GetImageFromArray(doc['mask'])
        for source in ['cine', 'optical_flow', 'registration_transform', 'diff']:
            try:
                img = doc[source]
                if source in ['optical_flow', 'registration_transform']:
                    img = np.sqrt(np.square(img[:, :, 0]) + np.square(img[:, :, 1]))
                image = sitk.GetImageFromArray(img)
                diagnostics, features = self.get_features(image, mask)
                doc['diagnostics'][source] = diagnostics
                doc['features'][source] = features
            except Exception as e:
                doc[f'error_{source}'] = str(e)
                doc[f'exception_{source}'] = traceback.format_exc()
