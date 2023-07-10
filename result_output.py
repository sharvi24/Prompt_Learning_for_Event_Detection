import numpy as np
import json

valid_dir = "./maven_data/valid.json"
train_dir = "./maven_data/train.json"

def load_data(input_dir):
    data_items = []

    with open(input_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        item = json.loads(line)
        data_items.append(item)
    return data_items

def get_vocab(train_dir, valid_dir):
    train_data = load_data(train_dir)
    valid_data = load_data(valid_dir)
    vocab = {"None": 0}
    for item in train_data:
        for event in item["events"]:
            if event[-1] not in vocab:
                vocab[event[-1]] = len(vocab)
    for item in valid_data:
        for event in item["events"]:
            if event[-1] not in vocab:
                vocab[event[-1]] = len(vocab)
    return vocab

vocabulary = get_vocab(train_dir, valid_dir)
inv_vocabulary = {v:k for k,v in vocabulary.items()}
test_preds = list(np.load("test.npy", allow_pickle=True))

with open("Output.txt", "w") as text_file:
    for i in range(len(test_preds)):
        labs = []
        for j in range(len(test_preds[i])):
            if inv_vocabulary[test_preds[i][j]] != 'None':
                labs.append(inv_vocabulary[test_preds[i][j]])
        text_file.write("{"+f"\'predictions\': {labs}"+"}\n")
