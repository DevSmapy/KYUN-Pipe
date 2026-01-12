import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Load data
# Using Kaggle input path for the competition datasets
train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

# 2. Feature Selection & Target Definition
# Drop columns that are unique identifiers or high-cardinality strings to prevent overfitting
# Note: 'Cabin' will be parsed later into meaningful features (deck/num/side
cabin = train["Cabin"]
X = train.drop(["PassengerId", "Name", "Cabin", "Transported"], axis=1)
y = train["Transported"]

# Identify numerical and categorical features for separate preprocessing
cat_columns = X.select_dtypes("object").columns
num_columns = X.select_dtypes("number").columns

# 3. Missing Value Imputation
# Numerical: Use K-Nearest Neighbors to estimate missing values based on similar rows
# Categorical: Use the most frequent value (mode) as a simple fallback
knn_imputer = KNNImputer()
simple_imputer = SimpleImputer(strategy="most_frequent")

# Transform and reconstruct DataFrames to maintain column names
num_filled = pd.DataFrame(
    knn_imputer.fit_transform(X[num_columns]), columns=num_columns
)
cat_filled = pd.DataFrame(
    simple_imputer.fit_transform(X[cat_columns]), columns=cat_columns
)

# concatenate all columns
X = pd.concat([num_filled, cat_filled, cabin, y], axis=1).dropna()
y = X["Transported"]

# split string (cabin)
cabin = X["Cabin"].str.split("/")
X["deck"] = cabin.str[0]
X["num"] = cabin.str[1].astype(int)
X["side"] = cabin.str[2]

# transform category columns by One-Hot-Encoding
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(
    transform="pandas"
)
cat_columns = X.select_dtypes("object").columns
num_columns = X.select_dtypes("number").columns

cat_preproc = onehotencoder.fit_transform(X[cat_columns])

X_preproc = pd.concat([cat_preproc, X[num_columns]], axis=1)

# hold out
train_X, valid_X, train_y, valid_y = train_test_split(
    X_preproc, y, test_size=0.3, random_state=1
)

# train random forest classifer
rf = RandomForestClassifier()
rf.fit(train_X, train_y)

# prediction
pred = rf.predict(valid_X)

f1score = f1_score(valid_y, pred)
accuracy = accuracy_score(valid_y, pred)
print(f"f1_score:{f1score}")
print(f"accuracy:{accuracy}")
