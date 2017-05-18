import tensorflow as tf
import numpy as np
from dict import Vocabulary
import sys, os

'''
make onehot vector 
for training

'''

def onehot(x,size):
    
    ret = np.zeros((size),dtype=float)
    ret[x-1] = 1.0

    return ret

'''
Text CNN RNN 
'''

class TextCNNRNN:
    def __init__(self,embedding_size,n_classes, rnn_size,sequence_length,embedding_mat,filter_sizes):
        
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size

        self.input_data = tf.placeholder(tf.int32,[None,1])
        self.target_data = tf.placeholder(tf.int32,[None,n_classes])

        self.seq_len = sequence_length

        self.batch_size = tf.placeholder(tf.int32,[None])
        
        self.real_len = tf.placeholder(tf.int32,[None])
        
        embedding = tf.Variable(embedding_mat)

        

        self.embedded_chars = tf.nn.embedding_lookup(embedding,self.input_data)



        emb = tf.expand_dims(self.embedded_chars,-1)


        

        




'''       
    def run_epoch(self):
        pass
        
    def train(self):
        pass

'''

