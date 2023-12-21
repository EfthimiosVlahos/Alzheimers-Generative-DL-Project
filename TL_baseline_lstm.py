import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
import helper_functions as hf

DATA_PATH = "data/alzheimersdata.txt"
NEW_DATA = "data/TL_processed_data.txt"
TRAIN_DATA = "data/TL_train_data.txt"
TEST_DATA = "data/TL_test_data.txt"
VAL_DATA = "data/TL_val_data.txt"

#init random seed
seed = 1
np.random.seed(seed)

"""Filter Dataset Into Smaller Segments"""
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        # Remove uncommon letters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line) and ('P' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))

# Access filtered data
max_seq_len = hf.max_sequence(NEW_DATA)
var = hf.padFile(NEW_DATA, max_seq_len)
print("Shape:", var.shape)
"""Load & Save Datasets"""
#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)

#split full train set into smaller train set and validation (dev) set
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)

# Save the data
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

# Transfer Learn Model
transferLearned_BaselineLSTM = Sequential()
baseline = keras.models.load_model('models/baseline_lstm.h5')

# Freeze layers
layer_index = 0
for layer in baseline.layers[:-1]:
    if layer_index < 2:
        layer.trainable = False
    
    transferLearned_BaselineLSTM.add(layer)
    layer_index += 1
    
transferLearned_BaselineLSTM.add(Dense(units = 34, activation='softmax'))
print(transferLearned_BaselineLSTM.layers)   
transferLearned_BaselineLSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = transferLearned_BaselineLSTM.fit(x_train, y_train_2, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])
path = 'models/TL_baseline_lstm.h5'
transferLearned_BaselineLSTM.save(path)
hf.generate_curves(history, 'graphs/TL_baseline_lstm.png')
