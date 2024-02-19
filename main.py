import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from pycimg import CImg
from radiomics import featureextractor
from simple_settings import settings

os.environ.setdefault('SIMPLE_SETTINGS', 'settings')


@dataclass
class Sample:
    label: int  # 0 - serce jest zdrowe, 1 - w o brębie miokarbium są blizny
    cine: np.ndarray  # oryginał - rozkurczone
    cine_delayed: np.ndarray  # ten sam obszar co cine, ale w trakcie 1/4 skurczu
    optical_flow: np.ndarray  # optical flow - przepływ z opencv
    registration_transform: np.ndarray  # registration - inna transformacja
    mask: np.ndarray  # gdzie jest ściana serca, wyznaczana automatem, 1 to nasz obszar do szukania


class Main:
    data: list[Sample] = []

    def __init__(self):
        if Path.exists(settings.CACHE_FILE):
            self.data = pickle.loads(settings.CACHE_FILE.read_bytes())
            print(f'File "{settings.CACHE_FILE}": {len(self.data)} samples')
        else:
            self.data = []
            for file in settings.DATA_DIR.iterdir():
                arr = np.load(file, allow_pickle=True)
                print(f'File "{file}": {arr.size} samples')
                for d in arr:
                    self.data.append(Sample(**d))
            settings.CACHE_FILE.write_bytes(pickle.dumps(self.data))

    def show_img(self, arr):
        img = CImg(arr)
        img.display()

    def stats(self):
        label = 0
        max_value = 0
        min_value = sys.maxsize
        print(f"Samples: {len(self.data)}")
        for d in self.data:
            label += d.label
            if max_value < d.cine.max():
                max_value = d.cine.max()
            if min_value > d.cine.min():
                min_value = d.cine.min()
        print(f'Min value: {min_value}')
        print(f'Max value: {max_value}')
        print(f'Label: {label}')

    def run(self):
        params = {
            'binWidth': 25,
            'normalize': True,
            'normalizeScale': 1,
            # 'resampledPixelSpacing': [1, 1],
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True,
            'enableParallel': True,
            # 'resegmentRange': None,
            'label': 1,
            'additionalInfo': True
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**params)

        for sample in self.data[:1]:
            image = sitk.GetImageFromArray(sample.cine)
            mask = sitk.GetImageFromArray(sample.mask)
            result = extractor.execute(image, mask)

            # Access the extracted features
            for feature_name in result.keys():
                print(f"{feature_name}: {result[feature_name]}")


if __name__ == '__main__':
    main = Main()
    main.stats()
    # main.run()
