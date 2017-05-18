# -*- coding: utf-8 -*-
import os 
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import gensim as gs
import pandas as pd
from dict import *

mecab = MeCab.Tagger("Choi")

def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25,0.25,300) 
    return word_embeddings
def pad_sentences(sentences,padding_word= -1,forced_sequence_length=None):
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = forced_sequence_length
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding < 0:
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word]*num_padding
        padded_sentences.append(sentence)
    return padded_sentences
def build_dict(vocab_name):
    #pass
    vocab = load_from_file(vocab_name)
    return vocab
def batch_iter(data,batch_size,num_epochs,shuffle = True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epochs = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arrange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]

def pad_data(data,seq_length):
    padded_data = []
    for i in range(len(data)):
        if len(data[i]) >= seq_length:
            dd = data[i][0:seq_length-11]
        else:
            dd = data[i] + [-1]* (seq_length - len(data[i]))

        padded_data.append(dd)
    return padded_data
                    
def load_data(filename,vocab_name):
    vocab = Vocabulary(filename)
    with open(filename,"rb") as f:
        lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        line = line.strip()

        data1 = []
        node = mecab.parseToNode(line)
        while node:
            word = node.surface
            node = node.next
            data1.append(vocab.stoi(word))
        data.append(data1)
           
    return data
        
if __name__=='__main__':
    data = load_data("data_use.txt","dict.vocab")
    data1 = pad_data(data,seq_length=20)
    
    print data1[0]
    print data[0]
