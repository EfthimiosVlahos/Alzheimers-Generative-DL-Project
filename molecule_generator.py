import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
import helper_functions as hf

#init random seed
seed = 1
np.random.seed(seed)

# Filter Dataset Into Smaller Segments
DATA_PATH = "data/data.txt"
NEW_DATA = "data/processed_data.txt"
TRAIN_DATA = 'data/train_data.txt'
TEST_DATA = 'data/test_data.txt'
VAL_DATA = 'data/val_data.txt'
ALZHEIMERS_DATA = 'data/alzheimersdata.txt'

# Filter Dataset Into Smaller Segments
count=0
new = open(NEW_DATA, "w")
for line in open(DATA_PATH, "r"):
    count += 1
    if 30 < len(line) < 50 and count < 300000:
        # Remove uncommon characters
        if ('T' not in line) and ('V' not in line) and ('g' not in line) and ('L' not in line) and ('8' not in line):
            new.write(line)
file_new = np.array(list(open(NEW_DATA)))
print(file_new.shape)

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

# Load the model and data
model = keras.models.load_model('models/TL_improved_lstm.h5')
ad_data = np.array(list(open(ALZHEIMERS_DATA)))
print(ad_data.shape)
smiles=[]

# Create 100 molecules
for i in range(100):
    if i % 10 == 0:
      print(i)
    start = np.random.randint(0, len(ad_data)-1)
    single_seed = ad_data[start][0:15]
    prediction = hf.generate_text(single_seed, 35, model, 50, tokenizer)
    print("Prediction",prediction)
    smiles.append(prediction)
print(smiles)

# Filter molecules
generated_molecules = hf.checkSMILES(smiles)
print(generated_molecules)
