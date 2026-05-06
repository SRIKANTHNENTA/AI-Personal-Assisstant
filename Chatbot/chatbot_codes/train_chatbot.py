import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import random
from nltk.stem import WordNetLemmatizer
import os

lemmatizer = WordNetLemmatizer()

# Load intents.json
with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore = ['?', '!', ',', "'s"]

# Preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern = doc[0]
    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

X_train = list(training[:, 0])
y_train = list(training[:, 1])

# Build Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile and Train
from keras.optimizers import Adam
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=10, verbose=1)

# Save Model
model.save('mymodel.h5')
print("✅ Success: Model trained and saved as mymodel.h5")
