import numpy as np
import helper_functions as hf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras_tuner import HyperModel
import keras_tuner

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


''' 
# CODE WE USED TO TUNE OUR LSTM

# create the model class
class Model(HyperModel): 
  def build(self, hp): 
    units1 = hp.Int('units1', min_value=32, max_value=512, step=32) 
    dropout1 = hp.Float(name="dropout1", min_value=0.0, max_value=0.3, step=0.05) 
    units2 = hp.Int('units2', min_value=32, max_value=512, step=32) 
    dropout2 = hp.Float(name="dropout2", min_value=0.0, max_value=0.3, step=0.05) 
    units3 = hp.Int('units3', min_value=32, max_value=512, step=32) 
    dropout3 = hp.Float(name="dropout3", min_value=0.0, max_value=0.3, step=0.05) 
    units4 = hp.Int('units4', min_value=32, max_value=512, step=32) 
    dropout4 = hp.Float(name="dropou4", min_value=0.0, max_value=0.3, step=0.05) 
    units5 = hp.Int('units5', min_value=32, max_value=512, step=32) 
    dropout5 = hp.Float(name="dropout5", min_value=0.0, max_value=0.3, step=0.05) 
    model = Sequential() 
    model.add(LSTM(units = units1, return_sequences= True, input_shape = (x_train.shape[1:]))) 
    model.add(Dropout(dropout1)) 
    model.add(LSTM(units = units2, return_sequences=True)) 
    model.add(Dropout(dropout2)) 
    model.add(LSTM(units = units3, return_sequences = True)) 
    model.add(Dropout(dropout3)) 
    model.add(LSTM(units = units4, return_sequences=True)) 
    model.add(Dropout(dropout4)) 
    model.add(LSTM(units = units5)) 
    model.add(Dropout(dropout5)) 
    model.add(Dense(units=43, activation = 'softmax')) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model 

# Create the tuner and set conditions
hypermodel = Model() 
tuner = keras_tuner.BayesianOptimization( 
                        hypermodel=hypermodel, 
                        objective = "val_accuracy", 
                        max_trials =3, 
                        overwrite=True, 
                        directory='BO_search_dir', 
                        project_name='better_lstm',
                        ) 
# Tune the model
tuner.search(x_train, y_train_2, epochs=3, validation_data=(x_val, y_val_2)) 
best_model = tuner.get_best_models()[0] 
best_model.build(input_shape= (x_train.shape[1:])) 
best_model.summary() 
print(tuner.results_summary()) 
'''
# Build the model
tuned_model = Sequential()
tuned_model.add(LSTM(units = 512, return_sequences= True, input_shape = (x_train.shape[1:])))
tuned_model.add(Dropout(0.0))
tuned_model.add(LSTM(units = 512, return_sequences= True))
tuned_model.add(Dropout(0.0))
tuned_model.add(LSTM(units = 512, return_sequences= True))
tuned_model.add(Dropout(0.0))
tuned_model.add(LSTM(units = 512, return_sequences= True))
tuned_model.add(Dropout(0.0))
tuned_model.add(LSTM(units = 512))
tuned_model.add(Dropout(0.0))
tuned_model.add(Dense(units = 43, activation='softmax'))
tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(tuned_model.summary())

# Train the LSTM Model
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
history = tuned_model.fit(x_train, y_train_2, epochs = 40, batch_size = 128, validation_data=(x_val, y_val_2), verbose=1, callbacks = [early]) 

# Save info
path = 'models/improved_lstm.h5' 
tuned_model.save(path) 
hf.generate_curves(history, 'graphs/improved_lstm.png')