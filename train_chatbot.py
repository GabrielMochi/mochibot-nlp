import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

intents = json.loads(open("intents.json").read())

for intent in intents:
    for pattern in intent["patterns"]:
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        documents.append((word, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [WordNetLemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []

    pattern_words = doc[0]
    pattern_words = [WordNetLemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

training_patterns = list(training[:, 0])
training_intents = list(training[:, 1])

print("training data created")

model = Sequential()

model.add(Dense(128, input_shape=(len(training_patterns[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(training_intents[0]), activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(
    np.array(training_patterns),
    np.array(training_intents),
    epochs=200,
    batch_size=5,
    verbose=1
)

model.save("chatbot_model.h5", hist)

print("model created")
