from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

lemmatizer = WordNetLemmatizer()

model = load_model("chatbot_model.h5")

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

ERROR_THRESHOLD = 0.25


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, show_details=True):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)

    for sentence_word in sentence_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i] = 1

                if show_details:
                    print("found in bag: %s" % word)

    return np.array(bag)


def predict_class(sentence):
    bag = bow(sentence, show_details=False)
    responses = model.predict(np.array([bag]))[0]
    results = [[i, response] for i, response in enumerate(responses) if response >= ERROR_THRESHOLD]
    return_list = []

    results.sort(key=lambda x: x[1], reverse=True)

    for result in results:
        return_list.append({
            "intent": classes[result[0]],
            "probability": str(result[1])
        })

    return return_list
