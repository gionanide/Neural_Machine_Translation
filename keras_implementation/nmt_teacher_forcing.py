#!usr/bin/python
#import tensorflow as tf
import re
import string
from unicodedata import normalize
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import nltk
import preprocessing as preprocessing
import baseline_model as baseline_model
import model1 as model1
import evaluate_model as evaluate_model


     
        
'''

our model is a small model, the complexity can be increased using a larger number of examples, length of phrases and increasing the size of the vocabulary

'''             
if __name__ == '__main__':

        path = 'deu.txt'
        
        text = preprocessing.readFile(path)
        
        pairs = preprocessing.makePairs(text)
        
        cleaned_words = preprocessing.cleaning(pairs)
        
        #visualize(cleaned_words)
        #print(cleaned_words.shape) --> (192881, 2) 0 -english || 1 -german
        
        
        cleaned_words = cleaned_words[:30000, :]
        
        np.random.shuffle(cleaned_words)

 
        #english preprocessing
        english_tokenizer = preprocessing.tokenizer(cleaned_words[:,0])
        english_vocabulary_size = len(english_tokenizer.word_index) + 1
        english_max_sentence_length, english_max_sentence = preprocessing.max_sentence_length(cleaned_words[:,0])
        
        
        #print(english_tokenizer.word_index)
        
        print('\n ----------------- Insights ----------------- \n')
        print('English vocabulary size:',english_vocabulary_size)
        print('English max length sentence:',english_max_sentence_length)
        print('English max sentence:',english_max_sentence)
        
        
        #german preprocessing
        german_tokenizer = preprocessing.tokenizer(cleaned_words[:,1])
        german_vocabulary_size = len(german_tokenizer.word_index) + 1
        german_max_sentence_length, german_max_sentence = preprocessing.max_sentence_length(cleaned_words[:,1])
        
        print('German vocabulary size:',german_vocabulary_size)
        print('German max length sentence:',german_max_sentence_length)
        print('German max sentence:',german_max_sentence)
        
        
        
        print('EOS - end of sequence english',english_tokenizer.word_index['eos'])
        print('SOS - start of sequence english',english_tokenizer.word_index['sos'])
        print('EOS german',german_tokenizer.word_index['eos'])
        print('SOS german',german_tokenizer.word_index['sos'])
              
        
        
        #split test and train data
        train = cleaned_words[:28000]
        test = cleaned_words[28000:]
        
        print('Train size:',len(train)) # -- 173.529
        print('Test size:',len(test)) # -- 19289
        
        '''
        #--------------------------------------------------------------------------------------------------------------------------------> ENGLISH TO GERMAN
        #prepare train data
        train_english_input = preprocessing.encode_sequences(english_tokenizer, train[:, 0])
        #print(train_english_input)
        
        train_english_input = preprocessing.pad_sequences(english_max_sentence_length, train_english_input) 
        #print(train_english_input)
        
        train_german_output = preprocessing.encode_sequences(german_tokenizer, train[:, 1])
        train_german_output = preprocessing.pad_sequences(german_max_sentence_length, train_german_output)
        
        #make the target as an one hot encoding
        train_german_output = preprocessing.oneHotEncoding(train_german_output, german_vocabulary_size)
        
        #and one for english
        #train_english_output = oneHotEncoding(train_english_input, english_vocabulary_size)
                
        #print(train_german_output)
        #print(train_german_output[0].shape)
        #print(train_german_output[0][0].shape)    
        
        
        #prepare test data
        test_english_input = preprocessing.encode_sequences(english_tokenizer, test[:, 0])
        test_english_input = preprocessing.pad_sequences(english_max_sentence_length, test_english_input)
        
        test_german_output = preprocessing.encode_sequences(german_tokenizer, test[:, 1])
        test_german_output = preprocessing.pad_sequences(german_max_sentence_length, test_german_output)
        
        #make the target as an one hot encoding
        test_german_output = preprocessing.oneHotEncoding(test_german_output, german_vocabulary_size)
        
        #and one for the english output
        #test_english_output = oneHotEncoding(test_english_input, english_vocabulary_size)
        
        #-----------------------------------------------------------> Set
        print('\n ----------------- Feeding sets ----------------- \n')
        print('Input train shape:',train_english_input.shape)
        print('Output train shape:',train_german_output.shape)
        print('Input test shape:',test_english_input.shape)
        print('Output test shape:',test_german_output.shape)
        
        
        
         #initialize the model
        #nmt_model = baseline_model.NMT(english_vocabulary_size, german_vocabulary_size, english_max_sentence_length, german_max_sentence_length, 256)
        nmt_model = model1.NMT(english_vocabulary_size, german_vocabulary_size, english_max_sentence_length, german_max_sentence_length, 256)
        
        #train the model
        trained_model = evaluate_model.compile_train(nmt_model, train_english_input, train_german_output, test_english_input, test_german_output)
        
        
        #evaluate model
        evaluate_model.model_speech_evaluation(trained_model, german_tokenizer, train_english_input, train, role='Train')
        
        evaluate_model.model_speech_evaluation(trained_model, german_tokenizer, test_english_input, test, role='Test')
        '''
        
        
        
        
        #------------------------------------------------------------------------------------------------------------------------------------> GERMAN TO ENGLISH
        #prepare train data
        train_english_output = preprocessing.encode_sequences(english_tokenizer, train[:, 0])
        #print(train_english_input)
        
        train_english_output = preprocessing.pad_sequences(english_max_sentence_length, train_english_output) 
        #print(train_english_input)
        
        train_german_input = preprocessing.encode_sequences(german_tokenizer, train[:, 1])
        train_german_input = preprocessing.pad_sequences(german_max_sentence_length, train_german_input)
        
        #make the target as an one hot encoding
        #train_english_output = preprocessing.oneHotEncoding(train_english_output, english_vocabulary_size)
        
        #and one for english
        #train_english_output = oneHotEncoding(train_english_input, english_vocabulary_size)
                
        #print(train_german_output)
        #print(train_german_output[0].shape)
        #print(train_german_output[0][0].shape)    
        
        
        #prepare test data
        test_english_output = preprocessing.encode_sequences(english_tokenizer, test[:, 0])
        test_english_output = preprocessing.pad_sequences(english_max_sentence_length, test_english_output)
        
        test_german_input = preprocessing.encode_sequences(german_tokenizer, test[:, 1])
        test_german_input = preprocessing.pad_sequences(german_max_sentence_length, test_german_input)
        
        #make the target as an one hot encoding
        #test_english_output = preprocessing.oneHotEncoding(test_english_output, english_vocabulary_size)
        
        #and one for the english output
        #test_english_output = oneHotEncoding(test_english_input, english_vocabulary_size)
        
        #-----------------------------------------------------------> Set
        print('\n ----------------- Feeding sets ----------------- \n')
        print('Input train shape:',train_german_input.shape)
        print('Output train shape:',train_english_output.shape)
        print('Input test shape:',test_german_input.shape)
        print('Output test shape:',test_english_output.shape)
        
        print('\n\n')
        encoder_inputs_train = train_german_input
        print('Encoder inputs:',encoder_inputs_train.shape)
        decoder_inputs_train = train_english_output
        
        #make a numpy arra with zeros because we want to add it as a column to the end, recall that we erase the first element
        zeros_train = np.zeros(len(train))
        #print('zeros column:',zeros_train.shape)
        zeros_train = np.reshape(zeros_train, (zeros_train.shape[0],1))
        #print('zeros column ready to add:',zeros_train.shape)
        

        print('Decoder inputs:',decoder_inputs_train.shape)
        print('Decoder input example:',decoder_inputs_train[0])
        target_outputs_train = train_english_output[:,1:english_max_sentence_length]
        target_outputs_train = np.concatenate((target_outputs_train, zeros_train), axis=1).astype(int)
        print('Target output:',target_outputs_train.shape)
        print('Target output example:',target_outputs_train[0])
        
        target_outputs_train = preprocessing.oneHotEncoding(target_outputs_train, english_vocabulary_size)
        
        
        print('\n\n')
        encoder_inputs_test = test_german_input
        print('Encoder inputs:',encoder_inputs_test.shape)
        decoder_inputs_test = test_english_output
        
        
        #make a numpy arra with zeros because we want to add it as a column to the end, recall that we erase the first element
        zeros_test = np.zeros(len(test))
        #print('zeros column:',zeros_train.shape)
        zeros_test = np.reshape(zeros_test, (zeros_test.shape[0],1))
        #print('zeros column ready to add:',zeros_train.shape)
        
        
        
        print('Decoder inputs:',decoder_inputs_test.shape)
        print('Decoder inputs:',decoder_inputs_test.shape)
        print('Decoder input example:',decoder_inputs_test[0])
        target_outputs_test = test_english_output[:,1:english_max_sentence_length]
        target_outputs_test = np.concatenate((target_outputs_test, zeros_test), axis=1).astype(int)
        print('Target output:',target_outputs_test.shape)
        print('Target output example:',target_outputs_test[0])
        
        target_outputs_test = preprocessing.oneHotEncoding(target_outputs_test, english_vocabulary_size)
        
        
        
        nmt_model = model1.NMT(german_vocabulary_size, english_vocabulary_size, german_max_sentence_length, english_max_sentence_length, 256)
        
        #train the model
        trained_model = evaluate_model.compile_train_seq2seq(nmt_model, encoder_inputs_train, decoder_inputs_train, target_outputs_train, encoder_inputs_test, decoder_inputs_test, target_outputs_test)
        
        
        #evaluate model
        evaluate_model.model_speech_evaluation(trained_model, english_tokenizer, train_german_input, train, role='Train')
        
        evaluate_model.model_speech_evaluation(trained_model, english_tokenizer, test_german_input, test, role='Test')
        
        
        
        
        
        
        
        
        
        
        
        
