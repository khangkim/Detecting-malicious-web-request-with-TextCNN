# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
#from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from keras.initializers import Constant
import csv

# Setting up parameters
maximum_features = 5000  # Maximum number of words to consider as features
maximum_length = 100  # Maximum length of input sequences
word_embedding_dims = 10  # Dimension of word embeddings
no_of_filters = 250  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the convolutional filters
hidden_dims = 250  # Number of neurons in the hidden layer
dropout_rate = 0.5  # Dropout rate for regularization
batch_size = 32  # Batch size for training
epochs = 2  # Number of training epochs
threshold = 0.5  # Threshold for binary classification

# Loading the dataset
x_train = []
y_train =[]
x_test =[]
y_test =[]

with open("./database/url_train.csv", mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        x_train.append(row[0])
        y_train.append(int(row[1]))

with open("./database/url_test.csv", mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        x_test.append(row[0])
        y_test.append(int(row[1]))


# Create a tokenizer
tokenizer = Tokenizer(num_words=maximum_features)
tokenizer.fit_on_texts(x_train)

# Convert text to sequences
x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)

# Pad sequences to ensure uniform length
x_train_padded = np.array(pad_sequences(x_train_sequences, maxlen=maximum_length, padding='post', truncating='post'))
x_test_padded = np.array(pad_sequences(x_test_sequences, maxlen=maximum_length, padding='post', truncating='post'))
y_train = np.array(y_train)
y_test = np.array(y_test)

# Prepare text for Word2Vec training
x_train_text = []
for line in x_train:
	x_train_text.append(line.split())

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=x_train_text, vector_size=word_embedding_dims, window=5, min_count=1, workers=4, sg=1)

# Create embedding matrix
embedding_matrix = np.zeros((maximum_features, word_embedding_dims))
for word, i in tokenizer.word_index.items():
    if i < maximum_features:
        try:
            embedding_vector = word2vec_model.wv[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

# Build the model
model = Sequential()

# Add embedding layer with pre-trained Word2Vec weights
embedding_layer = Embedding(maximum_features, word_embedding_dims, embeddings_initializer=Constant(embedding_matrix),
                            input_length=maximum_length, trainable=False)
model.add(embedding_layer)

# Add 1D convolutional layer with ReLU activation
model.add(Conv1D(no_of_filters, kernel_size, padding='valid', activation='relu', strides=1))

# Add global max pooling layer to reduce dimensionality
model.add(GlobalMaxPooling1D())

# Add dropout layer for regularization
model.add(Dropout(dropout_rate))

# Add dense hidden layer with ReLU activation
model.add(Dense(hidden_dims, activation='relu'))

# Add another dropout layer for regularization
model.add(Dropout(dropout_rate))

# Add output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train_padded, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test_padded, y_test))

# Predict probabilities for test data
y_pred_prob = model.predict(x_test_padded)

# Convert probabilities to binary classes based on threshold
y_pred = (y_pred_prob > threshold).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
