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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
data_path = 'datasetV2StringFormatted.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
chunk_size = 100000  # Adjust based on your memory capacity
chunks = []

for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    # Process each chunk as needed, for example appending it to a list
    chunks.append(chunk)
    print(f"Processed {len(chunks) * chunk_size} rows")
# Concatenate all chunks if you need the full DataFrame
dataframe = pd.concat(chunks, ignore_index=True)
dataframe["disease"] = dataframe["disease"].str.replace('"', '').str.replace("'", '')
print("a")
min_samples = 2 #only diseases with more than 2
dataframe = dataframe.groupby(dataframe["disease"]).filter(lambda x: len(x) >= min_samples)
#setting the columns we want, for now exactly the same but if we want to remove some for the picture version we can easily do it here
#X is input, y is output
X=dataframe["symptomstring"]
y=dataframe["disease"]
print("b")
#Split data into train(80%) and test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify=y)
print("c")
X_train = X_train.str.split('')
X_train = X_train.apply(lambda x: [int(i) for i in x[1:] if i != ''])
X_train = np.array(X_train.tolist())  # Convert to NumPy array
print("d")

X_test = X_test.str.split('')
X_test = X_test.apply(lambda x: [int(i) for i in x[1:] if i != ''])
X_test = np.array(X_test.tolist())  # Convert to NumPy array
print("e")
#creatting tree with a certain max depth(layers)
#class_dt = tree.DecisionTreeClassifier(max_depth=250)

def downcast_df(df):
    float_cols = df.select_dtypes(include=['float']).columns
    int_cols = df.select_dtypes(include=['int']).columns

    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    return df

# Apply downcasting to save memory

rf_clf = RandomForestClassifier(n_jobs=12, max_depth=200, verbose=1, n_estimators = 35, min_samples_split=2, min_samples_leaf=2)
rf_clf.fit(X_train, y_train)
print("f")
y_pred = rf_clf.predict(X_test)

new_accuracy = accuracy_score(y_test, y_pred) * 100
print("New Accuracy: ", new_accuracy)

joblib.dump(rf_clf, 'rfc_modelV4.pkl')
old_model = joblib.load('rfc_modelV4.pkl')
y_pred = old_model.predict(X_test)
old_accuracy = accuracy_score(y_test, y_pred) * 100
print("Old Accuracy: ", old_accuracy)

if new_accuracy > old_accuracy:
    joblib.dump(rf_clf, 'rfc_modelV4.pkl')
    print("New model saved")
else:
    print("Old model kept")
# y_pred = gbr.predict(X_test)

# print("Accuracy: ", metrics.accuracy_score(y_test, y_pred) * 100)