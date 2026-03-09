import os

import pandas as pd

if __name__ == "__main__":
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(filename)

    df_projects = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/projects_data.csv"
    )
    print(df_projects.head())
    print(df_projects.isnull().sum())
