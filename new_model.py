import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import top_k_categorical_accuracy

# Load your dataset
data = pd.read_csv("your_dataset.csv")  # Replace with your dataset file path
symptoms = data["symptoms"].values
diseases = data["diseases"].apply(lambda x: x.split(',')).values  # Assumes diseases are comma-separated

# Encode the labels as a binary matrix
mlb = MultiLabelBinarizer()
disease_labels = mlb.fit_transform(diseases)
num_classes = len(mlb.classes_)

# Tokenize the symptoms
max_words = 10000  # Define vocab size
max_len = 100      # Define max sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(symptoms)
symptom_sequences = tokenizer.texts_to_sequences(symptoms)
symptom_sequences_padded = pad_sequences(symptom_sequences, maxlen=max_len)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(symptom_sequences_padded, disease_labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="sigmoid")  # Sigmoid for multi-label classification
])

# Compile the model with top-3 accuracy as a metric
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", top_3_accuracy])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy, top3_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Top 3 Accuracy:", top3_acc)

# Predict the top 3 diseases for a new symptom input
def predict_top_3_diseases(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_len)
    predictions = model.predict(input_padded)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_diseases = [mlb.classes_[i] for i in top_3_indices]
    return top_3_diseases

# Example prediction
input_text = "fever, cough, sore throat"  # Replace with actual symptoms
print("Top 3 Predicted Diseases:", predict_top_3_diseases(input_text))
