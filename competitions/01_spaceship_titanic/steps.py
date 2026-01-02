import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder


class CabinSplitter(BaseEstimator, TransformerMixin):
    """Splits 'Cabin' into 'Deck', 'Num', and 'Side'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Cabin" in X.columns:
            X[["Deck", "Num", "Side"]] = X["Cabin"].str.split("/", expand=True)
            X["Num"] = pd.to_numeric(X["Num"], errors="coerce")
        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns from the DataFrame."""

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        existing_cols = [col for col in self.columns if col in X_out.columns]
        return X_out.drop(columns=existing_cols)


class MissingValueFiller(BaseEstimator, TransformerMixin):
    """Imputes missing values using the median for numerical features and the most frequent label for categorical features."""

    def fit(self, X, y=None):
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.simple_imputer = SimpleImputer(strategy="most_frequent")
        self.num_columns = X.select_dtypes("number").columns
        self.cat_columns = X.select_dtypes("object").columns
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_columns] = self.knn_imputer.fit_transform(X[self.num_columns])
        X[self.cat_columns] = self.simple_imputer.fit_transform(X[self.cat_columns])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ValueEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features.
    Supports OneHotEncoder and returns a DataFrame with proper column names.
    """

    def __init__(self, encoder=None):
        self.encoder = encoder or OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.feature_names = None

    def fit(self, X, y=None):
        X_out = X.copy()
        self.columns = X_out.select_dtypes("object").columns
        self.encoder.fit(X[self.columns])
        if hasattr(self.encoder, "get_feature_names_out"):
            self.feature_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X_out = X.copy()

        encoded_values = self.encoder.transform(X_out[self.columns])

        if self.feature_names is not None:
            # with OneHotEncoder
            encoded_df = pd.DataFrame(
                encoded_values, columns=self.feature_names, index=X_out.index
            )
        else:
            # with LabelEncoder
            encoded_df = pd.DataFrame(
                encoded_values, columns=self.columns, index=X_out.index
            )

        X_out = X_out.drop(columns=self.columns)
        return pd.concat([X_out, encoded_df], axis=1)
