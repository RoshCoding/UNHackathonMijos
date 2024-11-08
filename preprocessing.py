import pandas as pd

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

df = pd.read_csv("hf://datasets/dhivyeshrk/Disease-Symptom-Extensive-Clean/Final_Augmented_dataset_Diseases_and_Symptoms.csv")


df.head()

df = df[df["diseases"].str.len() < 20]
x = df.drop("diseases", axis=1)
y = df["diseases"]

x_new_cols = [i for i in list(x.columns) if len(i) < 20]
x_new_cols

x_new = x[x_new_cols]
x_new

y.value_counts()

disease_counts = y.value_counts()
disease_counts

dataframe = x_new.join(y)
dataframe

dataframe.to_csv("data.csv")