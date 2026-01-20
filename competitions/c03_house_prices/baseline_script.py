import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# split data by categorical, numeric
train = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)
df = train.copy()

cat_columns = df.select_dtypes("object").columns
num_columns = df.select_dtypes("number").columns

## impute categorical data (NaN is NA, so fillna "No")
df_cat = df[cat_columns].fillna("No")

## encode categorical data with LabelEncoder
label_encoder = LabelEncoder()
qual_cols = df_cat.filter(like="Qual").columns
for i in qual_cols:
    df_cat[i] = label_encoder.fit_transform(df_cat[i])

## encode categorical data with OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(
    transform="pandas"
)
diff_cols = cat_columns.difference(df_cat.filter(like="Qual").columns)
oh_encoded = onehotencoder.fit_transform(df_cat[diff_cols])

### concatenate
df_cat = pd.concat([oh_encoded, df_cat[qual_cols]], axis=1)


## impute numeric data by using KNN
knn_imputer = KNNImputer()
df_num = pd.DataFrame(knn_imputer.fit_transform(df[num_columns]), columns=num_columns)

final_df = pd.concat([df_cat, df_num], axis=1)

y = final_df["SalePrice"]
X = final_df.drop("SalePrice", axis=1)

train_X, valid_X, train_y, valid_y = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# train model
RF = RandomForestRegressor()
RF.fit(train_X, train_y)

# evaluate model
preds = RF.predict(valid_X)
np.sqrt(mean_squared_error(valid_y, preds))
