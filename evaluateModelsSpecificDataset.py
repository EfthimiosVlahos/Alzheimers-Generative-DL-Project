
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
        # Remove uncommon letters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line) and ('\ufeff' not in line) and ('P' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))
# print(file_new.shape)

"""Find Maximum Sequence Length"""

file = open(NEW_DATA)
max_seq_len = int(len(max(file,key=len)))
# print ("Max Sequence Length: ", max_seq_len)

var = hf.padFile(NEW_DATA, max_seq_len)
var = np.array(var)

"""Load & Save Datasets"""

#init random seed
seed = 1
np.random.seed(seed)
#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)

#split full train set into smaller train set and validation (dev) set
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)

# Save the data
np.savetxt(TRAIN_DATA, train, fmt ='%s', newline='')
np.savetxt(TEST_DATA, test, fmt ='%s', newline='')
np.savetxt(VAL_DATA, val, fmt='%s', newline='')

"""Test Data Processing"""
# Add uncommon missing to test
for x in train:
		if "5" in x:
		    test = np.append(test, x)

#Tokenize Data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, lower=False)

# Train Data Processing
train = hf.concatenate(train) 
x_train, y_train_2 = hf.prepare_data(train, tokenizer)

# Train Data Processing
test = hf.concatenate(test) 
x_test, y_test_2 = hf.prepare_data(test, tokenizer)

# Evaluate baseline
baseline = keras.models.load_model('models/TL_baseline_lstm.h5')
baseline_score = baseline.evaluate(x_test, y_test_2)
print(f"TL Baseline Test Loss: {baseline_score[0]}, TL Baseline Test Accuracy: {baseline_score[1]}")

# Evaluate improved
improved = keras.models.load_model('models/TL_improved_lstm.h5')
improved_score = improved.evaluate(x_test, y_test_2)
print(f"TL Improved Test Loss: {improved_score[0]}, TL Improved Test Accuracy: {improved_score[1]}")

# Evaluate hybrid
hybrid = keras.models.load_model('models/TL_hybrid.h5')
hybrid_score = hybrid.evaluate(x_test, y_test_2)
print(f"TL Hybrid Test Loss: {hybrid_score[0]}, TL Hybrid Test Accuracy: {hybrid_score[1]}")




