import joblib
import pandas as pd

# Load the saved model
model = joblib.load('rfc_modelV2.pkl')

df = pd.read_csv('evalData.csv')
first_row = df.drop("diseases", axis=1).iloc[0]
print(first_row)

# Make a prediction using the model
prediction = model.predict([first_row])
print(str(prediction))

