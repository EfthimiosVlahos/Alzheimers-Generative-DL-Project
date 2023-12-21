import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import helper_functions as hf
from tensorflow import keras

#init random seed
seed = 1
np.random.seed(seed)

DATA_PATH = "data/data.txt"
NEW_DATA = "data/processed_data.txt"
TRAIN_DATA = "data/train_data.txt"
TEST_DATA = "data/test_data.txt"
VAL_DATA = "data/val_data.txt"

"""Filter Dataset Into Smaller Segments"""
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):  
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line):
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


#split data into train/test
full_train, test = train_test_split(np.array(var), test_size=0.2, random_state=seed)
# full_train, test = train_test_split(np.array(var), test_size=0.25, random_state=seed)

#split full train set into smaller train set and validation (dev) set
# np.savetxt('/content/drive/MyDrive/CS230/full_train_data.txt', full_train, fmt ='%s', newline='')
train, val = train_test_split(np.array(full_train), test_size=0.10, random_state=seed)

# Save the data
np.savetxt(TRAIN_DATA, train, fmt ='%s', newline='')
np.savetxt(TEST_DATA, test, fmt ='%s', newline='')
np.savetxt(VAL_DATA, val, fmt='%s', newline='')


"""Data Processing"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, lower=False)

test = hf.concatenate(test) 
x_test, y_test_2 = hf.prepare_data(test, tokenizer)

# Baseline LSTM evaluation
baseline = keras.models.load_model('models/baseline_lstm.h5')
baseline_score = baseline.evaluate(x_test, y_test_2)
print(f"Baseline Test Loss: {baseline_score[0]}, Baseline Test Accuracy {baseline_score[1]}")

# Improved LSTM evaluation
improved = keras.models.load_model('models/improved_lstm.h5')
improved_score = improved.evaluate(x_test, y_test_2)
print(f"Improve Test Loss: {improved_score[0]}, Improved Test Accuracy {improved_score[1]}")

# Hybrid LSTM Evaluation
hybrid = keras.models.load_model('models/hybrid.h5')
hybrid_score = hybrid.evaluate(x_test, y_test_2)
print(f"Hybrid Test Loss: {hybrid_score[0]}, Hybrid Test Accuracy {hybrid_score[1]}")



