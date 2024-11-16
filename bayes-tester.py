import joblib
import numpy as np

# Load the model and TF-IDF vectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Transform new text data and make predictions
new_text = ["Im having stomach pain. I have a high temperature"]
new_text_tfidf = tfidf.transform(new_text)

# Get probabilities for each class
probabilities = model.predict_proba(new_text_tfidf)

# Get the indices of the top 3 highest probabilities
top3_indices = np.argsort(probabilities[0])[-5:][::-1]

# Get the class names (diagnoses) for the top 3 indices
top3_diagnoses = [model.classes_[index] for index in top3_indices]
top3_probabilities = [probabilities[0][index] for index in top3_indices]

# Print the top 3 diagnoses with their probabilities
for diagnosis, prob in zip(top3_diagnoses, top3_probabilities):
    print(f"Diagnosis: {diagnosis}, Probability: {prob:.2f}")
