import tensorflow as tf
from keras import layers, models, losses, optimizers
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


training_file = 'Ebrahim.txt'

raw_text = open(training_file, 'r').read()
raw_text = raw_text.lower()
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())


raw_text = " ".join(w for w in nltk.wordpunct_tokenize(raw_text)
         if w.lower() in words or not w.isalpha())
raw_text = ''.join([i for i in raw_text if not i.isdigit()])
encoded_string = raw_text.encode("ascii", "ignore")
raw_text = encoded_string.decode()
print(raw_text[:200])


bad_chars = ['~', '|', '\x11', '#', '$', '%','(', ')', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>','@', '\' , '^', '_', '`']

for i in bad_chars :
    raw_text = raw_text.replace(i, '')

print(raw_text[:200])

n_chars = len(raw_text)

chars = sorted(list(set(raw_text)))
n_vocab = len(chars)

index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
print(n_vocab)

seq_length = 160
n_seq = int(n_chars / seq_length)

X = np.zeros((n_seq, seq_length, n_vocab))
Y = np.zeros((n_seq, seq_length, n_vocab))

for i in range(n_seq):
	x_sequence = raw_text[i * seq_length : (i + 1) * seq_length]
	x_sequence_ohe = np.zeros((seq_length, n_vocab))
	for j in range(seq_length):
		char = x_sequence[j]
		index = char_to_index[char]
		x_sequence_ohe[j][index] = 1.
	X[i] = x_sequence_ohe
	y_sequence = raw_text[i * seq_length + 1 : (i + 1) * seq_length + 1]
	y_sequence_ohe = np.zeros((seq_length, n_vocab))
	for j in range(seq_length):
		char = y_sequence[j]
		index = char_to_index[char]
		y_sequence_ohe[j][index] = 1.
	Y[i] = y_sequence_ohe

print(X.shape)
print(Y.shape)

batch_size = 100
hidden_units = 600
n_epoch= 300
dropout = 0.1
learnng_rate = 0.001

tf.random.set_seed(42)

model = models.Sequential()
model.add(layers.LSTM(hidden_units, input_shape=(None, n_vocab), return_sequences=True, dropout=dropout))
model.add(layers.LSTM(hidden_units, return_sequences=True, dropout=dropout))
model.add(layers.TimeDistributed(layers.Dense(n_vocab, activation='softmax')))

optimizer = optimizers.RMSprop(lr=learnng_rate)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# print(model.summary())



def generate_text(model, gen_length, n_vocab, index_to_char):

    # Start with a randomly picked character
    index = np.random.randint(n_vocab)
    y_char = [index_to_char[index]]
    X = np.zeros((1, gen_length, n_vocab))
    for i in range(gen_length):
        X[0, i, index] = 1.
        indices = np.argmax(model.predict(X[:, max(0, i - 99):i + 1, :])[0], 1)
        index = indices[-1]
        y_char.append(index_to_char[index])
    return ''.join(y_char)


class ResultChecker(Callback):
    def __init__(self, model, N, gen_length):
        self.model = model
        self.N = N
        self.gen_length = gen_length

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:
            result = generate_text(self.model, self.gen_length, n_vocab, index_to_char)
            print('\nMy War and Peace:\n' + result)


filepath="weights2/weights_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1, mode='min')

result_checker = ResultChecker(model, 10, 500)

model.fit(X, Y, batch_size=batch_size, verbose=1, epochs=n_epoch,
                 callbacks=[result_checker, checkpoint, early_stop])