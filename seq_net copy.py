#@title Run this to load some packages and data! { display-mode: "form" }
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, Dropout
from keras.optimizers import Adam
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from scipy import sparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_path = "seqNet_modelV3.keras"
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

min_samples = 2 #only diseases with more than 2
dataframe = dataframe.groupby(dataframe["disease"]).filter(lambda x: len(x) >= min_samples)
dataframe = dataframe.groupby("disease").apply(lambda x: x.sample(n=20, random_state=1) if len(x) > 20 else x).reset_index(drop=True)

#setting the columns we want, for now exactly the same but if we want to remove some for the picture version we can easily do it here
#X is input, y is output
X=dataframe["symptomstring"]
y=dataframe["disease"]

#Split data into train(80%) and test(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 17, stratify=y)

X_train = X_train.str.split('')
X_train = X_train.apply(lambda x: [int(i) for i in x[1:] if i != ''])
print(X_train.head())
X_train = np.array(X_train.tolist())  # Convert to NumPy array


X_test = X_test.str.split('')
X_test = X_test.apply(lambda x: [int(i) for i in x[1:] if i != ''])
X_test = np.array(X_test.tolist())  # Convert to NumPy array
# Check if all rows in X_train have the same length
row_lengths_train = [len(row) for row in X_train]
print("Unique row lengths in X_train:", set(row_lengths_train))
# Check if all rows in X_test have the same length
row_lengths_test = [len(row) for row in X_test]
print("Unique row lengths in X_test:", set(row_lengths_test))
# X_train = X_train.str.split(',')
# X_test = X_test.str.split(',')
# print(X_train.head())
# print(y_train.head())
# print(X_test.head())
#Prepare X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


model = Sequential()

model.add(Dense(512, input_shape=(len(X_test[0]),), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train.unique()), activation='softmax'))


model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('newseq.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train_encoded, epochs=20, validation_data=(X_test, y_test_encoded), verbose=1,  callbacks=[checkpoint])

new_model = load_model('newseq.keras')

y_pred = new_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate accuracy with discrete predictions

new_accuracy = accuracy_score(y_test_encoded, y_pred)

print("Test Accuracy:", new_accuracy)

old_model = load_model(model_path)
y_pred = old_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

old_accuracy = accuracy_score(y_test_encoded, y_pred) 


print("old: ", old_accuracy, "new: ", new_accuracy)

if (new_accuracy > old_accuracy):
    new_model.save(model_path)
    print("MODEL UPDATED: old: ", old_accuracy, "new: ", new_accuracy)

# input_symptoms = ["headache", "sore throat", "fever", "runny nose"]

# symptomList = []
# # Create a binary array of the same length as X
# input_binary = [1 if symptom in input_symptoms else 0 for symptom in X]

# input = np.array(input_binary)
# input = np.reshape(input, (1, -1))

# predictions = load_model(model_path).predict(input)

# # Sort predictions to get the top 3
# top_3_indices = np.argsort(predictions[0])[-3:][::-1]
# top_3_values = predictions[0][top_3_indices]
# # Combine the probabilities with the class labels for easier interpretation
# for i, (index, prob) in enumerate(zip(top_3_indices, top_3_values)):
#     print(f"Prediction {i+1}: Class {dataframe[y][index]} with probability {prob:.2f}")
# Create and train our multi layer perceptron model two rows of 10 neurons for now
# nnet = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter= 10000)
# nnet.fit(X_train, y_train)

# # Predict what the classes are based on x he testing data
# predictions = nnet.predict(X_test)

# Print the score on the testing data
# print("MLP Testing Accuracy:")
# print(accuracy_score(y_test, predictions) * 100)
