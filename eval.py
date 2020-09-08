import locale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from joblib import Parallel
from joblib import delayed

import fungsi
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Menyiapkan format bahasa dan mengunduh dictionary
locale.setlocale(locale.LC_ALL, '')

random_seed = 42
epochs = 25
batch_size = 64

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# load model
model: tf.keras.models.Model = tf.keras.models.load_model('model/model.h5')
lb: LabelEncoder = joblib.load('model/labelencoder.joblib')
tk: tf.keras.preprocessing.text.Tokenizer = joblib.load('model/tokenizer.joblib')

# --- Data ingestion
# Baca CSV
df = pd.read_csv('dataset/eval.csv')

# --- preprocessing

df = df.dropna()
df = df.reset_index(drop=True)


print(df['sentiment'].value_counts())

# Banyaknya kelas
print(df['sentiment'].value_counts().plot(kind="bar"))

# pisahkan text dan target
targets = df["sentiment"].values
texts = df["text"].values


# preprocess text
texts = Parallel(n_jobs=4, verbose=10)(
    delayed(fungsi.preprocess_text)(text) for text in texts)

df_pre = pd.DataFrame({'text': texts, 'sentiment': targets})
df_pre.to_csv('dataset/eval_preprocessed.csv', index=None)

# word embedding
texts_tokenized = tk.texts_to_sequences(texts)
X_truth = tf.keras.preprocessing.sequence.pad_sequences(
    texts_tokenized, maxlen=200)

# Pisahkan data latih dan uji
y_truth = tf.keras.utils.to_categorical(lb.transform(targets))

# --- Evaluate
y_pred = model.predict(X_truth, batch_size=batch_size)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_truth, axis=1)

# Hitung statistik prediksi
print(classification_report(y_test, y_pred, target_names=lb.classes_))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=lb.classes_).plot(values_format='n')
plt.show()


df['pred'] = y_pred
df.to_csv('data1.csv')
