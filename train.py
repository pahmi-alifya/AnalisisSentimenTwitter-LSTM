import locale

import fungsi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from joblib import Parallel
from joblib import delayed

import nltk
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Menyiapkan format bahasa dan mengunduh dictionary
locale.setlocale(locale.LC_ALL, '')
nltk.download(['stopwords', 'punkt'])

random_seed = 42
epochs = 20
batch_size = 64

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# --- Data ingestion
# Baca CSV
df = pd.read_csv('dataset/train.csv')

# --- Preprocessing
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
df_pre.to_csv('dataset/train_preprocessed.csv', index=None)

# word embedding
tk = tf.keras.preprocessing.text.Tokenizer()
tk.fit_on_texts(texts)

train_tokenized = tk.texts_to_sequences(texts)
texts = tf.keras.preprocessing.sequence.pad_sequences(
    train_tokenized, maxlen=200)

# Pisahkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    texts, targets, test_size=0.20, random_state=random_seed)

# encode label
lb = LabelEncoder()
lb.fit(y_train)

y_train = tf.keras.utils.to_categorical(lb.transform(y_train))
y_test = tf.keras.utils.to_categorical(lb.transform(y_test))

# --- Training
# buat model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(
    X_train.max() + 1, 64, input_length=X_train.shape[1]))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# fitting data
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=epochs, batch_size=batch_size)

# plot history
fungsi.plot_history(history, epochs)

# --- Evaluation

y_pred = model.predict(X_test, batch_size=batch_size)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Hitung statistik prediksi
print(classification_report(y_test, y_pred, target_names=lb.classes_))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=lb.classes_).plot(values_format='n')
plt.show()


# --- save model
model.save('model/model.h5')
joblib.dump(tk, 'model/tokenizer.joblib')
joblib.dump(lb, 'model/labelencoder.joblib')


# df['predik'] = y_pred
# df.to_csv('data2.csv')
