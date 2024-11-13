import pandas as pd
import numpy as np
# Load the dataset
dataframe = pd.read_csv("datasetV2.csv")

# Extract diseases and symptoms
disease = dataframe["disease"]
symptomsLists = dataframe["symptoms"]

# Get all unique symptoms and set up the DataFrame
uniqueSymptoms = symptomsLists.str.split(',').explode().str.strip().str.replace('_', ' ').str.lower().unique()
print(len(uniqueSymptoms))
newDf = pd.DataFrame({'disease': disease, 'symptomString': ''}) 
print('a')

# Fill the 'disease' column
newDf["disease"] = disease
# Create a binary matrix for symptoms
def encode_symptoms(symptom_list):
    if pd.isna(symptom_list):  # Check if the symptom_list is NaN
        return ''.join(['0'] * len(uniqueSymptoms))
    symptom_set = set(s.strip().lower().replace('_', ' ') for s in symptom_list.split(','))
    return ''.join(['1' if symptom in symptom_set else '0' for symptom in uniqueSymptoms])

# Iterate over the DataFrame in chunks to avoid memory overload
chunk_size = 10000  # Process in chunks of 10,000 rows
for start_idx in range(0, len(symptomsLists), chunk_size):
    end_idx = min(start_idx + chunk_size, len(symptomsLists))
    chunk_symptoms = symptomsLists[start_idx:end_idx]
    
    # Apply the function and store the results for the current chunk
    encoded_symptoms = chunk_symptoms.apply(encode_symptoms).tolist()
    
    # Update the relevant rows of the new DataFrame
    newDf.iloc[start_idx:end_idx, 1:] = np.array(encoded_symptoms)
    print(f"Processed rows {start_idx} to {end_idx}")
# Clean column names
print('b')
newDf.columns = newDf.columns.str.strip().str.lower()

print('c')

# Save the DataFrame back to CSV
newDf.to_csv("datasetV2StringFormatted.csv", index=False)
print('d')