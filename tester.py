import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor

# Create a dictionary to store the parameters
params = {
    'binWidth': 25,
    'normalize': True,
    'normalizeScale': 1,
    'resampledPixelSpacing': [1, 1],
    'interpolator': 'sitkBSpline',
    'enableCExtensions': True,
    'enableParallel': True,
    'resegmentRange': [1, 100],
    'label': 1,
    'additionalInfo': True
}

# Create the feature extractor object with the specified parameters
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# Generate example image and mask (30x30)
image_np = np.random.rand(30, 30).astype(np.float64)  # Example image of type float64
mask_np = np.random.randint(0, 2, size=(30, 30), dtype=np.uint8)  # Example mask of type uint8

# Convert the ndarray image to a SimpleITK image
image_sitk = sitk.GetImageFromArray(image_np)

# Convert the ndarray mask to a SimpleITK image
mask_sitk = sitk.GetImageFromArray(mask_np)

# Create a dummy mask (all ones) as PyRadiomics requires a mask
# mask_sitk = sitk.Image(image_sitk.GetSize(), sitk.sitkUInt8)
# mask_sitk.CopyInformation(image_sitk)

# Extract features
result = extractor.execute(image_sitk, mask_sitk)

# Access the extracted features
for feature_name in result.keys():
    print(f"{feature_name}: {result[feature_name]}")
