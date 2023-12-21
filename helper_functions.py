from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs

seed = 1
np.random.seed(seed)

# reads the lines in a file
def read(fileName): 
        fileObj = open(fileName, "r") 
        words = fileObj.read().splitlines() 
        fileObj.close() 
        return words

# Opens file and returns a padded array to fit a specific size
def padFile(fileName, max_seq_len): 
	temp = read(fileName) 
	preprocessed_pad_text = [['?'] + list(i) for i in temp]  
	padded_text = pad_sequences(preprocessed_pad_text, dtype=object, maxlen=max_seq_len+1, padding="post", value="!") 
	var = ["".join(i) for i in padded_text] 
	print("Padded Strings: ", var[0:5]) 
	return var 

# Join inputed array into a single string
def concatenate(data): 
    res = ''.join(data) 
    return res 

# Print # of unique characters and total #
def data_breakdown(data):
	#Print Data Breakdown 
	n_chars = len(data) 
	n_vocab = len(list(set(data))) 
	print("# of Unique Characters:", n_chars) 
	print("# of Total Characters:", n_vocab) 

# Find the longest sequence in a file
def max_sequence(data):
	file = open(data)
	max_seq_len = int(len(max(file,key=len)))
	print ("Max Sequence Length: ", max_seq_len)
	return max_seq_len

# Use a sliding window to generate input and valid guesses for the input
def n_grams(seqLen, stepSize, data):

	input_chars = [] 
	next_char = [] 
	for i in range(0, len(data) - seqLen, stepSize): 
		input_chars.append(data[i : i + seqLen]) 
		next_char.append(data[i + seqLen]) 
	
	# Examples
	for i in range(5): 
		print("Input Sequence:", input_chars[i]) 
		print("Next Character Prediction:", next_char[i]) 

	return input_chars, next_char

# Generate the rest of the molecule given a seed text
def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):

	for _ in range(next_words):
		
		temp = []
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = np.array(token_list)

		# Resize
		for token in token_list:
			temp.append(token)
		token_list=np.array([temp])

		token_list = token_list[:15]

		predicted = model.predict(token_list, verbose=0)

		ind=np.argmax(predicted)

		output_word = ""
		for word,index in tokenizer.word_index.items():
			if index == ind:
				output_word = word
				break
			if output_word == "!":
				break

		seed_text += output_word
	return seed_text.title()

# Plot our graphs for a training of a model
def generate_curves(history, filename):
	fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

	axis1.plot(history.history["accuracy"], label='Train', linewidth=3)
	axis1.plot(history.history["val_accuracy"], label='Validation', linewidth=3)
	axis1.set_title('Model accuracy', fontsize=16, color="white")
	axis1.set_ylabel('accuracy')
	axis1.set_xlabel('epoch')
	axis1.legend(loc='lower right')

	axis2.plot(history.history["loss"], label='Train', linewidth=3)
	axis2.plot(history.history["val_loss"], label='Validation', linewidth=3)
	axis2.set_title('Model loss', fontsize=16, color="white")
	axis2.set_ylabel('loss')
	axis2.set_xlabel('epoch')
	axis2.legend(loc='upper right')
	plt.savefig(filename)
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



# Break our data down so it is usable by our model
def prepare_data(data, tokenizer):
	tokenizer.fit_on_texts(data) 
	new_data = tokenizer.texts_to_sequences(data) 
	data_breakdown(data)

	#N-Grams Sequence 
	seqLen = 15 
	stepSize = 1 
	input_chars, next_char = n_grams(seqLen, stepSize, new_data)
  
	#Assemble Validation Datasets 
	x_data = np.array(input_chars) 
	x_data.flatten() 
	y_data = np.array(next_char) 
	y_data_2 = to_categorical(y_data) 
	return x_data, y_data_2

# Checks our smiles and returns valid ones
def checkSMILES(smiles):
	valid_smiles = []
	for smile in smiles:
		new = ''
		for char in smile:
			if char != '?' and char != '!':	
				new += char

		if Chem.MolFromSmiles(smile, sanitize=False) is not None and len(smile) > 15:
			valid_smiles.append(smile)
			print(smile)

	return valid_smiles
