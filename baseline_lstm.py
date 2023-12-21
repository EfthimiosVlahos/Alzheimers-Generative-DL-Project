import numpy as np
import helper_functions as hf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

DATA_PATH = "data/data.txt"
NEW_DATA = "data/processed_data.txt"
TRAIN_DATA = 'data/train_data.txt'
TEST_DATA = 'data/test_data.txt'
VAL_DATA = 'data/val_data.txt'

#init random seed
seed = 1
np.random.seed(seed)

# Filter Dataset Into Smaller Segments
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        # Remove uncommon letters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))

# Access filtered data
max_seq_len = hf.max_sequence(NEW_DATA)
data = hf.padFile(NEW_DATA, max_seq_len) 
data = np.array(data) 

#split data into train/test
full_train, test = train_test_split(np.array(data), test_size=0.2, random_state=seed)

# split full train set into smaller train set and validation (dev) set
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)

#Save the data
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

# Build model
baselineLSTM = Sequential()
baselineLSTM.add(LSTM(units = 32, input_shape = (x_train.shape[1:])))
baselineLSTM.add(Dropout(0.2))
baselineLSTM.add(Dense(units = 43, activation='softmax'))
baselineLSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(baselineLSTM.summary())

# Train LSTM Model
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
history = baselineLSTM.fit(x_train, y_train_2, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])

# Save info
path = 'models/baseline_lstm.h5' 
baselineLSTM.save(path) 
hf.generate_curves(history, 'graphs/baseline_lstm.png')