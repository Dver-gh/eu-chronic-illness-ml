"""
Preprocessing subpackage: modular preprocessing components for ML pipeline.

Modules:
- missing_values: Handle NaN and missing data
- encoders: Categorical encoding (LabelEncoder, OneHotEncoder)
- scaling: Feature scaling (StandardScaler, MinMaxScaler)
- feature_engineering: Feature extraction and creation
"""

from .missing_values import (
    detect_missing_values,
    impute_missing_values,
    drop_rows_with_missing
)

from .encoders import (
    get_categorical_columns,
    encode_with_label_encoder,
    encode_with_one_hot,
    encode_with_sklearn_onehot,
    inverse_transform_labels
)

from .scaling import (
    get_numerical_columns,
    scale_features_standard,
    scale_features_minmax,
    get_scaling_statistics
)

from .feature_engineering import (
    extract_datetime_features,
    bin_continuous_feature,
    create_polynomial_features,
    create_interaction_features,
    create_ratio_features
)

__all__ = [
    'detect_missing_values',
    'impute_missing_values',
    'drop_rows_with_missing',
    'get_categorical_columns',
    'encode_with_label_encoder',
    'encode_with_one_hot',
    'encode_with_sklearn_onehot',
    'inverse_transform_labels',
    'get_numerical_columns',
    'scale_features_standard',
    'scale_features_minmax',
    'get_scaling_statistics',
    'extract_datetime_features',
    'bin_continuous_feature',
    'create_polynomial_features',
    'create_interaction_features',
    'create_ratio_features',
]
