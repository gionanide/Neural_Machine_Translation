#!usr/bin/python
import re
import string
from unicodedata import normalize
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import nltk



#read the file
def readFile(path):

        #open the file
        input_file = open(path,mode='r',encoding='UTF-8')
       
        #read the whole text
        whole_text = input_file.read()

        #close the file
        input_file.close()
        
        return whole_text
        


#------------------------------------------------------------------------------------> Preprocessing
def makePairs(text):

        #split the input text to pairs, after you erase the gaps from the edges
        lines = text.strip().split('\n')
        
        #make them pairs
        pairs = [line.split('\t') for line in lines]
        
        return pairs
        
       
       
'''

Cleaning prolicy:

---- remove all non printable chracters
---- remoce all the puncuation characters
---- make all the unicodes charaxters to ASCII
---- make all the letters lowercase
---- remove non alphabetic tokenss

'''
def cleaning(pairs):        
        
        #initialize a list which will contain the cleaned pairs
        cleaned_words = []
        
        #use the escape to backslash all the non alphanumeric characters
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        
        #remove pncuations using mapping
        table = str.maketrans('', '', string.punctuation)
        
        #iterate all the pairs
        for pair in pairs:
        
                #keep every pair
                cleaned_pair = []
                #print(pair)
                
                for sentence in pair:
                
                        #print(sentence)
                
                        #normalize unicode characters
                        sentence = normalize('NFD',sentence).encode('ascii','ignore')
                        
                        sentence = sentence.decode('UTF-8')
                        
                        sentence = sentence.split()
                         
                        #convert to lowercase
                        sentence = [word.lower() for word in sentence]
                        
                        #remove puncuations from each token
                        sentence = [word.translate(table) for word in sentence]
                        
                        #remove non printable chars from each token
                        sentence = [re_print.sub('',w) for w in sentence]
                        
                        #remove tokens with numbers
                        sentence = [word for word in sentence if word.isalpha()]
                        
                        cleaned_pair.append(' '.join(sentence))
                        
                        #print(sentence)
                        
                        
                cleaned_words.append(cleaned_pair)
                        
                        #break
                #print(cleaned_words)
                #break
                
        #return an array with all the words cleaned
        return np.array(cleaned_words)
        
        
#check the results
def visualize(cleaned_words):

        for pair in cleaned_words:
        
                print(pair[0]+' ------> '+pair[1])
        
                #break


#we user keras tokenizer to map words to integers
def tokenizer(pairs):

        #initalize an instance of the Tokenizer class
        tokenizer = keras.preprocessing.text.Tokenizer()
        
        tokenizer.fit_on_texts(pairs)
        
        return tokenizer
        
        
#find the length of the max sentence
def max_sentence_length(pairs):

        #max of a list and its index
        max_len = 0

        for pair in pairs:
        
                if(len(pair.split()) > max_len):
                
                        max_len = len(pair.split())
                        max_sentence = pair.split()


        return max_len, max_sentence
        
        
        
        
#encode sequences
def encode_sequences(tokenizer, pairs):        
        
        sequence = tokenizer.texts_to_sequences(pairs)
        
        return sequence
     
     
     
        
#apply zero padding to sequences
def pad_sequences(length, pairs):

        sequence = keras.preprocessing.sequence.pad_sequences(pairs, maxlen=length, padding='post')
        
        return sequence
        
        
        
        
#apply one hot encoding to the output
def oneHotEncoding(sequences, vocabulary_size):

        #inialize target list
        target_list = []

        #iterate all the elements
        for sequence in sequences:
        
                #one hot encoding
                encoded = keras.utils.to_categorical(sequence, num_classes=vocabulary_size) 
                
                target_list.append(encoded)
                
        #make it as a numpy array
        targets = np.array(target_list)
        
        targets = targets.reshape(sequences.shape[0], sequences.shape[1], vocabulary_size)
        
        return targets
     
        
     
    
    
           
