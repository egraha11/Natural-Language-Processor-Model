import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Flatten, Dropout, RNN, Embedding, Input
from keras.models import Sequential
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import TextVectorization
from keras.utils import to_categorical
import random 

text = (open("wonderland.txt", encoding="utf8").read()).lower()

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

res = text_to_word_sequence(text)


words = set(res)
vocab_size = len(words)

x=[]
y=[]
sequence_len = 4

for word in range(len(res)-sequence_len):
    x.append(res[word:word+sequence_len])
    y.append(res[word+sequence_len])
    

vectorize_layer = TextVectorization(max_tokens=vocab_size,
                                    split='whitespace',
                                    output_mode='int',
                                    output_sequence_length=sequence_len)

vectorize_layer.adapt(res)

x_encoded = []

for seq in x:
    #convert each sequence represesented as a list into a string
    seq = " ".join(seq)
    
    encoded = vectorize_layer(seq)
    x_encoded.append(encoded)

y_modified = []

for word in y:
    try:
        y_modified.append(vectorize_layer.get_vocabulary().index(word))
    except Exception as e:
        y_modified.append(vectorize_layer.get_vocabulary().index("[UNK]"))

y_modified = to_categorical(y_modified)


embedding_dim = 10

#create model

#convert list of tensorflow objects to numpy array 
x_encoded = np.array(x_encoded)

model = Sequential()


model.add(Input(shape=(1, x_encoded.shape[1])))


model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=4))


model.add(LSTM(100,input_shape=(4,10), return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(100))

model.add(Dropout(0.2))



model.add(Dense(y_modified.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(x_encoded, y_modified, epochs=10, batch_size=10)


import random 

test_len=5

generated = []

#generate random seed to begin testing
x = np.array(random.choice([num for num,text in enumerate(vectorize_layer.get_vocabulary())])).reshape(-1, 1)

for word in range(test_len):
    #reshape prediction so model excepts a single input value
    x_padded = pad_sequences(x, maxlen=4)
    
    preds = list(model.predict(x_padded, verbose=0)[0])
    
    index = preds.index(max(preds))
    
    #retreive predicted word
    generated.append(vectorize_layer.get_vocabulary()[index])
    
    #re-initialize next word that goes into the model
    x=index
    x=np.array(index).reshape(-1,1)   
    
#print generated text
print(generated)