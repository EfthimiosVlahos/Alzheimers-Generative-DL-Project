import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import helper_functions as hf
from keras_tuner import HyperModel

#init random seed
seed = 1
np.random.seed(seed)

DATA_PATH = "data/alzheimersdata.txt"
NEW_DATA = "data/TL_processed_data.txt"
TRAIN_DATA = "data/TL_train_data.txt"
TEST_DATA = "data/TL_test_data.txt"
VAL_DATA = "data/TL_val_data.txt"

"""Filter Dataset Into Smaller Segments"""
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        # Remove uncommon letters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line)and ('P' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))

"""Find Maximum Sequence Length"""
file = open(NEW_DATA)
max_seq_len = int(len(max(file,key=len)))

# Access data
var = hf.padFile(NEW_DATA, max_seq_len)
var = np.array(var)

#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)

#split full train set into smaller train set and validation (dev) set
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)

# Save data
np.savetxt(TRAIN_DATA, train, fmt ='%s', newline='')
np.savetxt(TEST_DATA, test, fmt ='%s', newline='')
np.savetxt(VAL_DATA, val, fmt='%s', newline='')

tokenizer = Tokenizer(num_words=None, char_level=True, lower=False) 
# Train Data Processing
train = hf.concatenate(train) 
x_train, y_train_2 = hf.prepare_data(train, tokenizer)

#Val Data Processing
val = hf.concatenate(val) 
x_val, y_val_2 = hf.prepare_data(val, tokenizer)
print(len(set(train)))

# Build Transfer Learn Model
tuned_LSTM = keras.models.load_model('models/improved_lstm.h5')
TL_tuned_LSTM = Sequential()
layer_index = 0
for layer in tuned_LSTM.layers[:-1]:
    if layer_index < 6:
        layer.trainable = False

    TL_tuned_LSTM.add(layer)
    layer_index += 1

print(TL_tuned_LSTM.layers)
TL_tuned_LSTM.add(Dense(units = 34, activation='softmax'))

# Transfer learn
TL_tuned_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = TL_tuned_LSTM.fit(x_train, y_train_2, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])

# Save model
path = 'models/TL_improved_lstm.h5'
TL_tuned_LSTM.save(path)
hf.generate_curves(history, 'graphs/TL_improved_lstm.png')
