import os
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup

os.system("pip install pycountry")
from pycountry import countries  # noqa: E402


def print_lines(n, filename):
    f = open(filename)
    for i in range(n):
        print(f.readlines())
    f.close()


if __name__ == "__main__":
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(filename)

    df_projects = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/projects_data.csv"
    )

    df_population = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.csv",
        skiprows=4,
    )

    print_lines(
        1,
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.json",
    )

    df_json = pd.read_json(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.json"
    )

    with open(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.xml"
    ) as f:
        soup = BeautifulSoup(f, "lxml")

    i = 0
    for record in soup.find_all("record"):
        i += 1
        for record in record.find_all("field"):
            print(record["name"], ":", record.text)
        print()
        if i == 5:
            break

    conn = sqlite3.connect(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.db"
    )

    pd.read_sql("SELECT * FROM population_data", conn)

    pd.read_sql(
        'SELECT "Country_Name", "Country_Code", "1960" FROM population_data', conn
    )

    url = "http://api.worldbank.org/v2/countries/br;cn;us;de/indicators/SP.POP.TOTL/?format=json&per_page=1000"

    r = requests.get(url)
    r.json()

    pd.DataFrame(r.json()[1])

    f = open(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/rural_population_percent.csv"
    )
    for i in range(10):
        line = f.readline()
        print("line: ", i, line)
    f.close()

    df_rural = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/rural_population_percent.csv",
        skiprows=4,
    )
    df_electricity = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/electricity_access_percent.csv",
        skiprows=4,
    )

    df_rural.drop(["Unnamed: 62"], axis=1, inplace=True)
    df_electricity.drop(["Unnamed: 62"], axis=1, inplace=True)

    df = pd.concat([df_rural, df_electricity], axis=1)
    print(df.head())

    df_indicator = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/population_data.csv",
        skiprows=4,
    )
    df_indicator.drop(["Unnamed: 62"], axis=1, inplace=True)

    # read in the projects data set with all columns type string
    df_projects = pd.read_csv(
        "/kaggle/input/datasets/nilaychauhan/world-bank-datasets/projects_data.csv",
        dtype=str,
    )
    df_projects.drop(["Unnamed: 56"], axis=1, inplace=True)

    print(df_indicator[["Country Name", "Country Code"]].drop_duplicates())

    df_projects["Official Country Name"] = (
        df_projects["countryname"].str.split(";").str.get(0)
    )

    countries.get(name="Spain")

    countries.lookup("Kingdom of Spain")
