#@title Create the model { display-mode: "both" }
from sklearn import tree
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

data_path = 'data.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

#Split data into train(80%) and test(20%)
train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

# X_train should contain all columns except the 'diseases' column
X_train = train_df.drop('diseases', axis=1)
# y_train should contain the 'diseases' column
y_train = train_df['diseases']

X_test = test_df.drop('diseases', axis=1)
y_test = test_df['diseases']

#creatting tree with a certain max depth(layers)
#class_dt = tree.DecisionTreeClassifier(max_depth=250)

def downcast_df(df):
    float_cols = df.select_dtypes(include=['float']).columns
    int_cols = df.select_dtypes(include=['int']).columns

    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    return df

# Apply downcasting to save memory
dataframe = downcast_df(dataframe)

rf_clf = RandomForestClassifier(n_jobs=-1, max_depth=250, verbose=1, n_estimators = 30)
rf_clf.fit(X_train, y_train)
print(cross_val_score(rf_clf, X_train, y_train, cv=3).mean())

# y_pred = gbr.predict(X_test)

# print("Accuracy: ", metrics.accuracy_score(y_test, y_pred) * 100)

# Save the model to the outputs directory for capture
model_file = 'model.pkl'
joblib.dump(rf_clf, model_file)