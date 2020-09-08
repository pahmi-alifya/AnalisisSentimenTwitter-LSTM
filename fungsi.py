import re
import string

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import matplotlib.pyplot as plt

stopwords_list = set(nltk.corpus.stopwords.words(
    'english') + list(string.punctuation) + ['AT_USER', 'URL'])
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',
                  'URL', text)  # hapus URL
    text = re.sub(r'https[a-zA-Z0-9]+', '', text)
    text = re.sub(r'<[\w\d\D]+>', '', text)  # hapus simbol aneh
    text = re.sub(r'@[^\s]+', 'AT_USER', text)  # hapus mention
    text = re.sub(r'#[\w]+', '', text)  # hapus hashtag
    text = re.sub(r'[\d]+', '', text)  # hapus angka
    text = re.sub(r'&[\w]+', " ", text)  # hapus simbol aneh
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()  # removed white space
    text = nltk.tokenize.word_tokenize(text)  # tokenize
    # stopwords removal
    text = [word for word in text if word not in stopwords_list]
    text = [stemmer.stem(word) for word in text]  # stemming
    return ' '.join(text)


def plot_history(h, epochs):
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), h.history["loss"], label="Train")
    plt.plot(np.arange(0, epochs), h.history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="lower left")

    ax2 = plt.subplot(1, 2, 2, sharex=ax1)
    plt.plot(np.arange(0, epochs), h.history["accuracy"], label="Train")
    plt.plot(np.arange(0, epochs),
             h.history["val_accuracy"], label="Validation")
    ax2.set_xlabel("Epoch #")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower left")

    plt.show()
