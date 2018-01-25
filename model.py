import pandas as pd
import numpy as np

from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

data = ['This is a test document one','This is a test document two','well this document can be a cool one', 'another document for poor people like me','this is soo cool']
label = ['US CITE 1','US CITE 2']
target = [1, 2, 2, 0, 1]

# X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.2, random_state=42)

encoder = LabelEncoder()
encoder.fit(target)
encoded_Y = encoder.transform(target)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = np_utils.to_categorical(encoded_Y)

print(y_train)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
ELEMS, VOCAB = X_train_counts.shape



#this is logireg
# model = Sequential()
# model.add(Dense(1, activation='sigmoid', input_dim=X_train_tfidf.shape[1]))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train_tfidf.todense(), y_train, epochs=1, validation_data=(X_test,y_test))

model = Sequential()
model.add(Dense(100, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_tfidf.todense(), y_train, epochs=1)


# model = Sequential()
# model.add(Embedding(VOCAB, 128, input_length=300))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

