import tensorflow as tf
from keras.datasets import imdb
from keras import layers, models, losses, optimizers
from keras.utils import pad_sequences

vocab_size = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
maxlen = 200

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

tf.random.set_seed(42)
embedding_size = 32
model = models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_size))
model.add(layers.LSTM(50, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(50, dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = optimizers.Adam(lr=0.003)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 10
batch_size = 64

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epoch,
          validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose=0)[1]

print('Test accuracy with stacked LSTM:', acc)

model = models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_size))
model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True, dropout=0.2)))
model.add(layers.Bidirectional(layers.LSTM(50, dropout=0.2)))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

optimizer = optimizers.Adam(lr=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 10
batch_size = 64

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epoch,
          validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose=0)[1]

print('Test accuracy with stacked bidirectional LSTM:', acc)