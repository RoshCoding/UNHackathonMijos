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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_path = "seqNet_modelV2.keras"
data_path = 'dataset.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)


#setting the columns we want, for now exactly the same but if we want to remove some for the picture version we can easily do it here
dataframe = dataframe.groupby('diseases').apply(lambda x: x.sample(n=20, random_state=1) if len(x) > 20 else x).reset_index(drop=True)

#X is input, y is output
X = ["anxiety and nervousness","depression","shortness of breath","depressive or psychotic symptoms","sharp chest pain","dizziness","insomnia","abnormal involuntary movements","chest tightness","palpitations","irregular heartbeat","breathing fast","hoarse voice","sore throat","difficulty speaking","cough","nasal congestion","throat swelling","diminished hearing","lump in throat","throat feels tight","difficulty in swallowing","skin swelling","retention of urine","groin mass","leg pain","hip pain","suprapubic pain","blood in stool","lack of growth","emotional symptoms","elbow weakness","back weakness","pus in sputum","symptoms of the scrotum and testes","swelling of scrotum","pain in testicles","flatulence","pus draining from ear","jaundice","mass in scrotum","white discharge from eye","irritable infant","abusing alcohol","fainting","hostile behavior","drug abuse","sharp abdominal pain","feeling ill","vomiting","headache","nausea","diarrhea","vaginal itching","vaginal dryness","painful urination","involuntary urination","pain during intercourse","frequent urination","lower abdominal pain","vaginal discharge","blood in urine","hot flashes","intermenstrual bleeding","hand or finger pain","wrist pain","hand or finger swelling","arm pain","wrist swelling","arm stiffness or tightness","arm swelling","hand or finger stiffness or tightness","wrist stiffness or tightness","lip swelling","toothache","abnormal appearing skin","skin lesion","acne or pimples","dry lips","facial pain","mouth ulcer","skin growth","eye deviation","diminished vision","double vision","cross-eyed","symptoms of eye","pain in eye","eye moves abnormally","abnormal movement of eyelid","foreign body sensation in eye","irregular appearing scalp","swollen lymph nodes","back pain","neck pain","low back pain","pain of the anus","pain during pregnancy","pelvic pain","impotence","infant spitting up","vomiting blood","regurgitation","burning abdominal pain","restlessness","symptoms of infants","wheezing","peripheral edema","neck mass","ear pain","jaw swelling","mouth dryness","neck swelling","knee pain","foot or toe pain","bowlegged or knock-kneed","ankle pain","bones are painful","knee weakness","elbow pain","knee swelling","skin moles","knee lump or mass","weight gain","problems with movement","knee stiffness or tightness","leg swelling","foot or toe swelling","heartburn","smoking problems","muscle pain","infant feeding problem","recent weight loss","problems with shape or size of breast","underweight","difficulty eating","scanty menstrual flow","vaginal pain","vaginal redness","vulvar irritation","weakness","decreased heart rate","increased heart rate","bleeding or discharge from nipple","ringing in ear","plugged feeling in ear","itchy ear(s)","frontal headache","fluid in ear","neck stiffness or tightness","spots or clouds in vision","eye redness","lacrimation","itchiness of eye","blindness","eye burns or stings","itchy eyelid","feeling cold","decreased appetite","excessive appetite","excessive anger","loss of sensation","focal weakness","slurring words","symptoms of the face","disturbance of memory","paresthesia","side pain","fever","shoulder pain","shoulder stiffness or tightness","shoulder weakness","arm cramps or spasms","shoulder swelling","tongue lesions","leg cramps or spasms","abnormal appearing tongue","ache all over","lower body pain","problems during pregnancy","spotting or bleeding during pregnancy","cramps and spasms","upper abdominal pain","stomach bloating","changes in stool appearance","unusual color or odor to urine","kidney mass","swollen abdomen","symptoms of prostate","leg stiffness or tightness","difficulty breathing","rib pain","joint pain","muscle stiffness or tightness","pallor","hand or finger lump or mass","chills","groin pain","fatigue","abdominal distention","regurgitation.1","symptoms of the kidneys","melena","flushing","coughing up sputum","seizures","delusions or hallucinations","shoulder cramps or spasms","joint stiffness or tightness","pain or soreness of breast","excessive urination at night","bleeding from eye","rectal bleeding","constipation","temper problems","coryza","wrist weakness","eye strain","hemoptysis","lymphedema","skin on leg or foot looks infected","allergic reaction","congestion in chest","muscle swelling","pus in urine","abnormal size or shape of ear","low back weakness","sleepiness","apnea","abnormal breathing sounds","excessive growth","elbow cramps or spasms","feeling hot and cold","blood clots during menstrual periods","absence of menstruation","pulling at ears","gum pain","redness in ear","fluid retention","flu-like syndrome","sinus congestion","painful sinuses","fears and phobias","recent pregnancy","uterine contractions","burning chest pain","back cramps or spasms","stiffness all over","muscle cramps, contractures, or spasms","low back cramps or spasms","back mass or lump","nosebleed","long menstrual periods","heavy menstrual flow","unpredictable menstruation","painful menstruation","infertility","frequent menstruation","sweating","mass on eyelid","swollen eye","eyelid swelling","eyelid lesion or rash","unwanted hair","symptoms of bladder","irregular appearing nails","itching of skin","hurts to breath","nailbiting","skin dryness, peeling, scaliness, or roughness","skin on arm or hand looks infected","skin irritation","itchy scalp","hip swelling","incontinence of stool","foot or toe cramps or spasms","warts","bumps on penis","too little hair","foot or toe lump or mass","skin rash","mass or swelling around the anus","low back swelling","ankle swelling","hip lump or mass","drainage in throat","dry or flaky scalp","premenstrual tension or irritability","feeling hot","feet turned in","foot or toe stiffness or tightness","pelvic pressure","elbow swelling","elbow stiffness or tightness","early or late onset of menopause","mass on ear","bleeding from ear","hand or finger weakness","low self-esteem","throat irritation","itching of the anus","swollen or red tonsils","irregular belly button","swollen tongue","lip sore","vulvar sore","hip stiffness or tightness","mouth pain","arm weakness","leg lump or mass","disturbance of smell or taste","discharge in stools","penis pain","loss of sex drive","obsessions and compulsions","antisocial behavior","neck cramps or spasms","pupils unequal","poor circulation","thirst","sleepwalking","skin oiliness","sneezing","bladder mass","knee cramps or spasms","premature ejaculation","leg weakness","posture problems","bleeding in mouth","tongue bleeding","change in skin mole size or color","penis redness","penile discharge","shoulder lump or mass","polyuria","cloudy eye","hysterical behavior","arm lump or mass","nightmares","bleeding gums","pain in gums","bedwetting","diaper rash","lump or mass of breast","vaginal bleeding after menopause","infrequent menstruation","mass on vulva","jaw pain","itching of scrotum","postpartum problems of the breast","eyelid retracted","hesitancy","elbow lump or mass","muscle weakness","throat redness","joint swelling","tongue pain","redness in or around nose","wrinkles on skin","foot or toe weakness","hand or finger cramps or spasms","back stiffness or tightness","wrist lump or mass","skin pain","low back stiffness or tightness","low urine output","skin on head or neck looks infected","stuttering or stammering","problems with orgasm","nose deformity","lump over jaw","sore in nose","hip weakness","back swelling","ankle stiffness or tightness","ankle weakness","neck weakness"] # Add your features!
y = 'diseases'


min_samples = 2 #only diseases with more than 2
dataframe = dataframe.groupby(y).filter(lambda x: len(x) >= min_samples)

#Split data into train(80%) and test(20%)
train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1,  stratify=dataframe[y])

#Prepare X_train, X_test, y_train, and y_test variables by extracting the appropriate columns:
X_train = train_df[X]
y_train = train_df[y]

X_test = test_df[X]
y_test = test_df[y]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

model = Sequential()

model.add(Dense(256, input_shape=(len(X),), activation='relu'))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

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

input_symptoms = ["headache", "sore throat", "fever", "runny nose"]

# Create a binary array of the same length as X
input_binary = [1 if symptom in input_symptoms else 0 for symptom in X]

input = np.array(input_binary)
input = np.reshape(input, (1, -1))

predictions = load_model(model_path).predict(input)

# Sort predictions to get the top 3
top_3_indices = np.argsort(predictions[0])[-3:][::-1]
top_3_values = predictions[0][top_3_indices]
# Combine the probabilities with the class labels for easier interpretation
for i, (index, prob) in enumerate(zip(top_3_indices, top_3_values)):
    print(f"Prediction {i+1}: Class {dataframe[y][index]} with probability {prob:.2f}")
# Create and train our multi layer perceptron model two rows of 10 neurons for now
# nnet = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter= 10000)
# nnet.fit(X_train, y_train)

# # Predict what the classes are based on x he testing data
# predictions = nnet.predict(X_test)

# Print the score on the testing data
# print("MLP Testing Accuracy:")
# print(accuracy_score(y_test, predictions) * 100)