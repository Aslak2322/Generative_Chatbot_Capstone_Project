import numpy as np
import re

data_path = "./human_chat.txt"

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

#Filter empty lines
lines = [line for line in lines if line.strip()]

input_docs = []
target_docs = []

input_tokens = set()
target_tokens = set()

#Split text into input/target
for i in range(0, len(lines) - 1, 2):
    input_doc = lines[i].replace("Human 1: ", "", 1)
    target_doc = lines[i + 1].replace("Human 2: ", "", 1)

    input_tokenized = re.findall(r"[\w']+|[^\s\w]", input_doc)
    target_tokenized = re.findall(r"[\w']+|[^\s\w]", target_doc)
    target_tokenized = ['<START>'] + target_tokenized + ['<END>']

    input_docs.append(input_tokenized)
    target_docs.append(target_tokenized)

    input_tokens.update(input_tokenized)
    target_tokens.update(target_tokenized)

#sort
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

#max token length
max_encoder_seq_length = max(len(doc) for doc in input_docs)
max_decoder_seq_length = max(len(doc) for doc in target_docs)

#dict creation
input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = {i: token for token, i in input_features_dict.items()}
reverse_target_features_dict = {i: token for token, i in target_features_dict.items()}

#matrix creation
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

#assigning one to the correct word
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(input_doc):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1
    
    for timestep, token in enumerate(target_doc):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep-1, target_features_dict[token]] = 1.



print(len(input_tokens))
print(len(target_tokens))
