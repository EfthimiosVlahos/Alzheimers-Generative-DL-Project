import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import helper_functions as hf
from tensorflow import keras

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
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line) and ('P' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))

"""Find Maximum Sequence Length"""

file = open(NEW_DATA)
max_seq_len = int(len(max(file,key=len)))
print ("Max Sequence Length: ", max_seq_len)


var = hf.padFile(NEW_DATA, max_seq_len)
var = np.array(var)

"""Load & Save Datasets"""

#init random seed
seed = 1
np.random.seed(seed)
#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)
# full_train, test = train_test_split(np.array(var), test_size=0.25, random_state=seed)

#split full train set into smaller train set and validation (dev) set
# np.savetxt('/content/drive/MyDrive/CS230/full_train_data.txt', full_train, fmt ='%s', newline='')
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)
np.savetxt(TRAIN_DATA, train, fmt ='%s', newline='')
np.savetxt(TEST_DATA, test, fmt ='%s', newline='')
np.savetxt(VAL_DATA, val, fmt='%s', newline='')

"""Train Data Processing"""

#Load Data (optional: only if previous cells are not run and data is saved already)
# train = pd.read_fwf('/content/drive/MyDrive/CS230/train_data.txt')

"""Train Data Processing"""

#Load Data (optional: only if previous cells are not run and data is saved already)
# train = pd.read_fwf('/content/drive/MyDrive/CS230/train_data.txt')

tokenizer = Tokenizer(num_words=None, char_level=True, lower=False) 

# Train Data Processing
train = hf.concatenate(train) 
x_train, y_train_2 = hf.prepare_data(train, tokenizer)

#Val Data Processing
val = hf.concatenate(val) 
x_val, y_val_2 = hf.prepare_data(val, tokenizer)


# Transfer Learn Model
hybrid = keras.models.load_model('models/hybrid.h5')
TL_Hybrid = Sequential()
layer_index = 0
print(len(hybrid.layers))

# Build layers
for layer in hybrid.layers[:-1]:
    print(layer)
    if layer_index < 6:
        layer.trainable = False

    TL_Hybrid.add(layer)
    layer_index += 1

TL_Hybrid.add(Dense(units = 34, activation='softmax'))

# Train the model
TL_Hybrid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = TL_Hybrid.fit(x_train, y_train_2, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early])

# Save info
path = 'models/TL_hybrid.h5'
TL_Hybrid.save(path)
hf.generate_curves(history, 'graphs/TL_Hybrid.png')