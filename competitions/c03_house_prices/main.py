import logging
import sys

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

module_path = "/kaggle/input/pipe-core"
if module_path not in sys.path:
    sys.path.append(module_path)

from DataLoader import DataLoader  # noqa: E402
from Preprocessor import DataSplitter, UniversalPreprocessor  # noqa: E402
from Transformers import OrdinalMapper  # noqa: E402

logger = logging.getLogger(__name__)

SCALE_QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
SCALE_EXPOSURE = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
SCALE_FINISH = {
    "None": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}

# apply_quality_value_encoding이 실제로 지원하는 ordinal 컬럼 정의(핵심 수정 포인트)
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "ExterQual": SCALE_QUAL,
    "ExterCond": SCALE_QUAL,
    "BsmtQual": SCALE_QUAL,
    "BsmtCond": SCALE_QUAL,
    "HeatingQC": SCALE_QUAL,
    "KitchenQual": SCALE_QUAL,
    "FireplaceQu": SCALE_QUAL,
    "GarageQual": SCALE_QUAL,
    "GarageCond": SCALE_QUAL,
    "PoolQC": SCALE_QUAL,
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
    "BsmtExposure": SCALE_EXPOSURE,
    "BsmtFinType1": SCALE_FINISH,
    "BsmtFinType2": SCALE_FINISH,
    "CentralAir": {"N": 0, "Y": 1},
}


class NoneFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna("None")


if __name__ == "__main__":
    # 1. Load and get data
    DATA_DIR = "/kaggle/input/house-prices-advanced-regression-techniques"
    dataloader = DataLoader(
        train_path=f"{DATA_DIR}/train.csv", test_path=f"{DATA_DIR}/test.csv"
    )
    dataloader.load_data()
    train, test = dataloader.get_data()

    # 2. Split data into train and validation sets(for protection dataleakage)
    train_set, valid_set = DataSplitter.split_random(train, test_size=0.3)

    # 3. Selection ordinal features
    ordinal_features = pd.Index(
        [c for c in train_set.select_dtypes("object").columns if c in ORDINAL_MAPS]
    )
    # 4. Set up preprocessing pipeline
    steps = [
        ("none_filler", NoneFiller()),
        ("ordinal_encoder", OrdinalMapper(ORDINAL_MAPS)),
    ]

    # 5. Run preprocessing pipeline
    preprocessor = UniversalPreprocessor(steps)
    train_X, test_X = preprocessor.run(train_set, test)
    valid_X, _ = preprocessor.run(valid_set, None)
