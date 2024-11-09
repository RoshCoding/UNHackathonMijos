# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = 'text_dataset.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Split the dataset into features and labels
X = data['text']
y = data['lebel_text']  # Make sure this column name matches your dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train the Multinomial Naive Bayes model
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model_nb.predict(X_test_tfidf)

# Calculate and print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the trained model and vectorizer
joblib.dump(model_nb, 'naive_bayes_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
