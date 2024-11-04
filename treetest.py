#@title Create the model { display-mode: "both" }
from sklearn import tree
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# This is the name of our data file, which was downloaded in the set up cell.
# Check out the file explorer (folder on the left toolbar) to see where that lives!
data_path = 'cancer.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

# Redefine `dataframe` to include only the columns discussed
dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
print (dataframe)
# Define a new, more descriptive `diagnosis_cat` column
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})
# We'll first specify what model we want, in this case a decision tree
class_dt = tree.DecisionTreeClassifier(max_depth=6)

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)
X = ['perimeter_mean', 'texture_mean', 'concavity_mean', 'area_mean', 'smoothness_mean', 'symmetry_mean'] # Add your features!
y = 'diagnosis'

# 1. Split data into train and test
train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

# 2. Prepare your X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:
X_train = train_df[X]
y_train = train_df[y]

X_test = test_df[X]
y_test = test_df[y]

# We use our previous `X_train` and `y_train` sets to build the model
class_dt.fit(X_train, y_train)

plt.figure(figsize=(13,8))  # set plot size
tree.plot_tree(class_dt, fontsize=10)

y_pred = class_dt.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred))

