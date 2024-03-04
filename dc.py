from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Result:
    prep: str
    method: str
    params: dict
    accuracy: float
    roc_auc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    random_state: int
    source: str = None


@dataclass
class Sample:
    label: int  # 0 - serce jest zdrowe, 1 - w o brębie miokarbium są blizny
    cine: np.ndarray  # oryginał - rozkurczone
    cine_delayed: np.ndarray  # ten sam obszar co cine, ale w trakcie 1/4 skurczu
    optical_flow: np.ndarray  # optical flow - przepływ z opencv
    registration_transform: np.ndarray  # registration - inna transformacja
    mask: np.ndarray  # gdzie jest ściana serca, wyznaczana automatem, 1 to nasz obszar do szukania
    mask_from_LGE: np.ndarray


@dataclass
class Feature:
    source: str
    diagnostics_Versions_PyRadiomics: str
    diagnostics_Versions_Numpy: str
    diagnostics_Versions_SimpleITK: str
    diagnostics_Versions_PyWavelet: str
    diagnostics_Versions_Python: str
    diagnostics_Configuration_Settings: dict
    diagnostics_Configuration_EnabledImageTypes: dict
    diagnostics_Image_original_Hash: str
    diagnostics_Image_original_Dimensionality: str
    diagnostics_Image_original_Spacing: tuple
    diagnostics_Image_original_Size: tuple
    diagnostics_Image_original_Mean: np.float64
    diagnostics_Image_original_Minimum: np.float64
    diagnostics_Image_original_Maximum: np.float64
    diagnostics_Mask_original_Hash: str
    diagnostics_Mask_original_Spacing: tuple
    diagnostics_Mask_original_Size: tuple
    diagnostics_Mask_original_BoundingBox: tuple
    diagnostics_Mask_original_VoxelNum: int
    diagnostics_Mask_original_VolumeNum: int
    diagnostics_Mask_original_CenterOfMassIndex: tuple
    diagnostics_Mask_original_CenterOfMass: tuple
    original_firstorder_10Percentile: np.ndarray
    original_firstorder_90Percentile: np.ndarray
    original_firstorder_Energy: np.ndarray
    original_firstorder_Entropy: np.ndarray
    original_firstorder_InterquartileRange: np.ndarray
    original_firstorder_Kurtosis: np.ndarray
    original_firstorder_Maximum: np.ndarray
    original_firstorder_MeanAbsoluteDeviation: np.ndarray
    original_firstorder_Mean: np.ndarray
    original_firstorder_Median: np.ndarray
    original_firstorder_Minimum: np.ndarray
    original_firstorder_Range: np.ndarray
    original_firstorder_RobustMeanAbsoluteDeviation: np.ndarray
    original_firstorder_RootMeanSquared: np.ndarray
    original_firstorder_Skewness: np.ndarray
    original_firstorder_TotalEnergy: np.ndarray
    original_firstorder_Uniformity: np.ndarray
    original_firstorder_Variance: np.ndarray
    original_glcm_Autocorrelation: np.ndarray
    original_glcm_ClusterProminence: np.ndarray
    original_glcm_ClusterShade: np.ndarray
    original_glcm_ClusterTendency: np.ndarray
    original_glcm_Contrast: np.ndarray
    original_glcm_Correlation: np.ndarray
    original_glcm_DifferenceAverage: np.ndarray
    original_glcm_DifferenceEntropy: np.ndarray
    original_glcm_DifferenceVariance: np.ndarray
    original_glcm_Id: np.ndarray
    original_glcm_Idm: np.ndarray
    original_glcm_Idmn: np.ndarray
    original_glcm_Idn: np.ndarray
    original_glcm_Imc1: np.ndarray
    original_glcm_Imc2: np.ndarray
    original_glcm_InverseVariance: np.ndarray
    original_glcm_JointAverage: np.ndarray
    original_glcm_JointEnergy: np.ndarray
    original_glcm_JointEntropy: np.ndarray
    original_glcm_MCC: np.ndarray
    original_glcm_MaximumProbability: np.ndarray
    original_glcm_SumAverage: np.ndarray
    original_glcm_SumEntropy: np.ndarray
    original_glcm_SumSquares: np.ndarray
    original_gldm_DependenceEntropy: np.ndarray
    original_gldm_DependenceNonUniformity: np.ndarray
    original_gldm_DependenceNonUniformityNormalized: np.ndarray
    original_gldm_DependenceVariance: np.ndarray
    original_gldm_GrayLevelNonUniformity: np.ndarray
    original_gldm_GrayLevelVariance: np.ndarray
    original_gldm_HighGrayLevelEmphasis: np.ndarray
    original_gldm_LargeDependenceEmphasis: np.ndarray
    original_gldm_LargeDependenceHighGrayLevelEmphasis: np.ndarray
    original_gldm_LargeDependenceLowGrayLevelEmphasis: np.ndarray
    original_gldm_LowGrayLevelEmphasis: np.ndarray
    original_gldm_SmallDependenceEmphasis: np.ndarray
    original_gldm_SmallDependenceHighGrayLevelEmphasis: np.ndarray
    original_gldm_SmallDependenceLowGrayLevelEmphasis: np.ndarray
    original_glrlm_GrayLevelNonUniformity: np.ndarray
    original_glrlm_GrayLevelNonUniformityNormalized: np.ndarray
    original_glrlm_GrayLevelVariance: np.ndarray
    original_glrlm_HighGrayLevelRunEmphasis: np.ndarray
    original_glrlm_LongRunEmphasis: np.ndarray
    original_glrlm_LongRunHighGrayLevelEmphasis: np.ndarray
    original_glrlm_LongRunLowGrayLevelEmphasis: np.ndarray
    original_glrlm_LowGrayLevelRunEmphasis: np.ndarray
    original_glrlm_RunEntropy: np.ndarray
    original_glrlm_RunLengthNonUniformity: np.ndarray
    original_glrlm_RunLengthNonUniformityNormalized: np.ndarray
    original_glrlm_RunPercentage: np.ndarray
    original_glrlm_RunVariance: np.ndarray
    original_glrlm_ShortRunEmphasis: np.ndarray
    original_glrlm_ShortRunHighGrayLevelEmphasis: np.ndarray
    original_glrlm_ShortRunLowGrayLevelEmphasis: np.ndarray
    original_glszm_GrayLevelNonUniformity: np.ndarray
    original_glszm_GrayLevelNonUniformityNormalized: np.ndarray
    original_glszm_GrayLevelVariance: np.ndarray
    original_glszm_HighGrayLevelZoneEmphasis: np.ndarray
    original_glszm_LargeAreaEmphasis: np.ndarray
    original_glszm_LargeAreaHighGrayLevelEmphasis: np.ndarray
    original_glszm_LargeAreaLowGrayLevelEmphasis: np.ndarray
    original_glszm_LowGrayLevelZoneEmphasis: np.ndarray
    original_glszm_SizeZoneNonUniformity: np.ndarray
    original_glszm_SizeZoneNonUniformityNormalized: np.ndarray
    original_glszm_SmallAreaEmphasis: np.ndarray
    original_glszm_SmallAreaHighGrayLevelEmphasis: np.ndarray
    original_glszm_SmallAreaLowGrayLevelEmphasis: np.ndarray
    original_glszm_ZoneEntropy: np.ndarray
    original_glszm_ZonePercentage: np.ndarray
    original_glszm_ZoneVariance: np.ndarray
    original_ngtdm_Busyness: np.ndarray
    original_ngtdm_Coarseness: np.ndarray
    original_ngtdm_Complexity: np.ndarray
    original_ngtdm_Contrast: np.ndarray
    original_ngtdm_Strength: np.ndarray

    @property
    def values(self):
        return {k: v for k, v in asdict(self).items() if k != 'source' and not k.startswith('diagnostics')}
