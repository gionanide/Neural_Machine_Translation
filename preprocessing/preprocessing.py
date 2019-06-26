#!usr/bin/python
import re
import string
from unicodedata import normalize
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import tensorflow as tf



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
def cleaning(pairs,forcing):        
        
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
                        
                        if forcing:
                        
                                #add the begin and stop symbols in the begging and in the end of each sentence
                                sentence.insert(0,'<SOS>')
                                sentence.append('<EOS>')
                        
                        #print(sentence)
                        
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
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        
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

        sequence = tf.keras.preprocessing.sequence.pad_sequences(pairs, maxlen=length, padding='post')
        
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
        
        
        
        
#remove Start Of Sequence symbol 
def removeSOS(array,set_size):

        #make an array of zeros same length as our input array
        zeros = np.zeros(set_size)
        
        #reshape it
        zeros = np.reshape(zeros, (zeros.shape[0], 1))
        
        #take all the ements except from the first one (Start Of Sequence)
        array = array[:,1:]
        
        #concat a column of zeros to the end the keep the same padding
        array = np.concatenate((array,zeros), axis=1).astype(int)
        
        #return array
        return array
        
        
        
#remoce End Of Sequence sumbol       
def removeEOS(array):
        
        #where the array value is 2 replace is it with 0
        array[array==2] = 0
                
        
        return array
        
        
        
        
def format_data(path, dataset_length, train_per_cent, translation_flow):

        #define if you want to use teacher forcing
        teacher_forcing = False

        
        text = readFile(path)
        
        pairs = makePairs(text)
        
        cleaned_words = cleaning(pairs,forcing=teacher_forcing)
        
        #visualize(cleaned_words)
        #print(cleaned_words.shape) --> (192881, 2) 0 -english || 1 -german
        
        
        cleaned_words = cleaned_words[:dataset_length, :]
        
        np.random.shuffle(cleaned_words)

 
        #english preprocessing
        english_tokenizer = tokenizer(cleaned_words[:,0])
        english_vocabulary_size = len(english_tokenizer.word_index) + 1
        english_max_sentence_length, english_max_sentence = max_sentence_length(cleaned_words[:,0])
        
        
        #print(english_tokenizer.word_index)
        
        print('\n ----------------- Insights ----------------- \n')
        print('English vocabulary size:',english_vocabulary_size)
        print('English max length sentence:',english_max_sentence_length)
        print('English max sentence:',english_max_sentence)
        
        
        #german preprocessing
        german_tokenizer = tokenizer(cleaned_words[:,1])
        german_vocabulary_size = len(german_tokenizer.word_index) + 1
        german_max_sentence_length, german_max_sentence = max_sentence_length(cleaned_words[:,1])
        
        print('German vocabulary size:',german_vocabulary_size)
        print('German max length sentence:',german_max_sentence_length)
        print('German max sentence:',german_max_sentence)
        
        
        
        #print('EOS - end of sequence english',english_tokenizer.word_index['eos'])
        ##print('SOS - start of sequence english',english_tokenizer.word_index['sos'])
        #print('EOS german',german_tokenizer.word_index['eos'])
        #print('SOS german',german_tokenizer.word_index['sos'])
              
              
        train_length = int(dataset_length * train_per_cent)
        
        
        #split test and train data
        train = cleaned_words[:train_length]
        test = cleaned_words[train_length:]
        
        print('Train size:',len(train)) # -- 173.529
        print('Test size:',len(test)) # -- 19289
        
        
        #-----------------------------> Training sets
        #prepare train data
        english_train = encode_sequences(english_tokenizer, train[:, 0])
        #print(train_english_input)

        english_train = pad_sequences(english_max_sentence_length, english_train) 
        #print(train_english_input)

        german_train = encode_sequences(german_tokenizer, train[:, 1])
        german_train = pad_sequences(german_max_sentence_length, german_train)
        
        
        #-----------------------------> Test sets
        #prepare test data
        english_test = encode_sequences(english_tokenizer, test[:, 0])
        english_test = pad_sequences(english_max_sentence_length, english_test)


        german_test = encode_sequences(german_tokenizer, test[:, 1])
        german_test = pad_sequences(german_max_sentence_length, german_test)
        
        
        
        #--------------------------------------------------------------------------------------------------------------------------------> ENGLISH TO GERMAN
        if (translation_flow=='etg'):
        
                print('\n------------------------- English to German translation -------------------------\n')

                #make the target as an one hot encoding
                german_train = oneHotEncoding(german_train, german_vocabulary_size)


                #make the target as an one hot encoding
                german_test = oneHotEncoding(german_test, german_vocabulary_size)

                #and one for the english output
                #test_english_output = oneHotEncoding(test_english_input, english_vocabulary_size)

                #assign the values as encoder and decoder inputs, because we are going from english to german, encoder inputs are English and decoder outputs are German
                encoder_input_train = english_train
                
                decoder_output_train = german_train
                
                encoder_input_test = english_test
                
                decoder_output_test = german_test
                
                #and some other properties for the training
                input_vocabulary_size = english_vocabulary_size
                
                input_tokenizer = english_tokenizer
                
                input_max_sentence_length = english_max_sentence_length
                
                output_vocabulary_size = german_vocabulary_size
                
                output_tokenizer = german_tokenizer
                
                output_max_sentence_length = german_max_sentence_length
                
                
                
                
        #--------------------------------------------------------------------------------------------------------------------------------> GERMAN TO ENGLISH
        elif (translation_flow=='gte'):
        
                print('\n------------------------- German to English translation -------------------------\n')

                #make the target as an one hot encoding
                english_train = oneHotEncoding(english_train, english_vocabulary_size)


                #make the target as an one hot encoding
                english_test = oneHotEncoding(english_test, english_vocabulary_size)

                #and one for the english output
                #test_english_output = oneHotEncoding(test_english_input, english_vocabulary_size)
                
                
                #assign the values as encoder and decoder inputs, because we are going from english to german, encoder inputs are German and decoder outputs are English
                encoder_input_train = german_train
                
                decoder_output_train = english_train
                
                encoder_input_test = german_test
                
                decoder_output_test = english_test
                
                #and some other properties for the training
                input_vocabulary_size = german_vocabulary_size
                
                input_tokenizer = german_tokenizer
                
                input_max_sentence_length = german_max_sentence_length
                
                output_vocabulary_size = english_vocabulary_size
                
                output_tokenizer = english_tokenizer
                
                output_max_sentence_length = english_max_sentence_length
                
     
       
        return encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test, input_vocabulary_size, input_tokenizer, input_max_sentence_length, output_vocabulary_size, output_tokenizer, output_max_sentence_length, train, test
        
        
        
        
        
        

#different format for teacher forcing procedure
def teacher_forcing_format(path, dataset_length, train_per_cent, translation_flow):

        #define if you want to use teacher forcing
        teacher_forcing = True

        text = readFile(path)
        
        pairs = makePairs(text)
        
        cleaned_words = cleaning(pairs,forcing=teacher_forcing)
        
        #visualize(cleaned_words)
        #print(cleaned_words.shape) --> (192881, 2) 0 -english || 1 -german
        
        
        cleaned_words = cleaned_words[:dataset_length, :]
        
        np.random.shuffle(cleaned_words)

 
        #english preprocessing
        english_tokenizer = tokenizer(cleaned_words[:,0])
        english_vocabulary_size = len(english_tokenizer.word_index) + 1
        english_max_sentence_length, english_max_sentence = max_sentence_length(cleaned_words[:,0])
        
        
        #print(english_tokenizer.word_index)
        
        print('\n ----------------- Insights ----------------- \n')
        print('English vocabulary size:',english_vocabulary_size)
        print('English max length sentence:',english_max_sentence_length)
        print('English max sentence:',english_max_sentence)
        
        
        #german preprocessing
        german_tokenizer = tokenizer(cleaned_words[:,1])
        german_vocabulary_size = len(german_tokenizer.word_index) + 1
        german_max_sentence_length, german_max_sentence = max_sentence_length(cleaned_words[:,1])
        
        print('German vocabulary size:',german_vocabulary_size)
        print('German max length sentence:',german_max_sentence_length)
        print('German max sentence:',german_max_sentence)
        
        
        
        #print('EOS - end of sequence english',english_tokenizer.word_index['eos'])
        ##print('SOS - start of sequence english',english_tokenizer.word_index['sos'])
        #print('EOS german',german_tokenizer.word_index['eos'])
        #print('SOS german',german_tokenizer.word_index['sos'])
              
              
        train_length = int(dataset_length * train_per_cent)
        
        
        #split test and train data
        train = cleaned_words[:train_length]
        test = cleaned_words[train_length:]
        
        print('Train size:',len(train)) # -- 173.529
        print('Test size:',len(test)) # -- 19289
        
        
        #-----------------------------> Training sets
        #prepare train data
        english_train = encode_sequences(english_tokenizer, train[:, 0])
        #print(train_english_input)

        english_train = pad_sequences(english_max_sentence_length, english_train) 
        #print(train_english_input)

        german_train = encode_sequences(german_tokenizer, train[:, 1])
        german_train = pad_sequences(german_max_sentence_length, german_train)
        
        
        #-----------------------------> Test sets
        #prepare test data
        english_test = encode_sequences(english_tokenizer, test[:, 0])
        english_test = pad_sequences(english_max_sentence_length, english_test)


        german_test = encode_sequences(german_tokenizer, test[:, 1])
        german_test = pad_sequences(german_max_sentence_length, german_test)
        
        
        #--------------------------------------------------------------------------------------------------------------------------------> ENGLISH TO GERMAN
        if (translation_flow=='etg'):
        
                print('\n------------------------- English to German translation -------------------------\n')

                #assign the values as encoder and decoder inputs, because we are going from english to german, encoder inputs are English and decoder outputs are German
                encoder_input_train = english_train
                
                decoder_input_train = german_train

                decoder_output_train = german_train

                encoder_input_test = english_test
                
                decoder_input_test = german_test
                
                decoder_output_test = german_test
                
   
                #and some other properties for the training
                input_vocabulary_size = english_vocabulary_size
                
                input_tokenizer = english_tokenizer
                
                input_max_sentence_length = english_max_sentence_length
                
                output_vocabulary_size = german_vocabulary_size
                
                output_tokenizer = german_tokenizer
                
                output_max_sentence_length = german_max_sentence_length
                
                
                
                
        #--------------------------------------------------------------------------------------------------------------------------------> GERMAN TO ENGLISH
        elif (translation_flow=='gte'):
        
                print('\n------------------------- German to English translation -------------------------\n')
                
                
                #assign the values as encoder and decoder inputs, because we are going from english to german, encoder inputs are German and decoder outputs are English
                encoder_input_train = german_train
                
                decoder_input_train = english_train
                
                decoder_output_train = english_train
                
                encoder_input_test = german_test
                
                decoder_input_test = english_test
                
                decoder_output_test = english_test
                
                #and some other properties for the training
                input_vocabulary_size = german_vocabulary_size
                
                input_tokenizer = german_tokenizer
                
                input_max_sentence_length = german_max_sentence_length
                
                output_vocabulary_size = english_vocabulary_size
                
                output_tokenizer = english_tokenizer
                
                output_max_sentence_length = english_max_sentence_length
                
                
        
        
        #--------------------------------------------------------------------> Encoder inputs
        print('\n\n')
        print('Encoder format:  "sentence"<EOS>')
        encoder_input_train = np.copy(encoder_input_train)
        
        encoder_input_train = removeSOS(encoder_input_train,len(train))

        print('Encoder inputs train:',encoder_input_train.shape)
        print('Encoder inputs train example:',encoder_input_train[0])
        
        
        encoder_input_test = np.copy(encoder_input_test)
        
        encoder_input_test = removeSOS(encoder_input_test,len(test))
        
        print('Encoder inputs test:',encoder_input_test.shape)
        print('Encoder inputs test example:',encoder_input_test[0])
        print('\n\n')
        
        
        
        #-------------------------------------------------------------------------> Decoder inputs
        print('\n\n')
        print('Decoder format:  <SOS>"sentence"')
        decoder_input_train = np.copy(decoder_input_train)
        
        decoder_input_train = removeEOS(decoder_input_train)

  
        print('Decoder inputs train:',decoder_input_train.shape)
        print('Decoder input train example:',decoder_input_train[0])


        decoder_input_test = np.copy(decoder_input_test)
        
        decoder_input_test = removeEOS(decoder_input_test)
        

        print('Decoder inputs test:',decoder_input_test.shape)
        print('Decoder input example test:',decoder_input_test[0])
        print('\n\n')
        
        
        #----------------------------------------------------------------------------------------------------> Target outputs
        print('\n\n')
        print('Target format:  "sentence"<EOS>')
        target_output_train = np.copy(decoder_input_train)
        
        target_output_train = removeSOS(target_output_train,len(train))
        
        #target_outputs_train = preprocessing.oneHotEncoding(np.copy(target_outputs_train), english_vocabulary_size)

        print('Target output:',target_output_train.shape)
        print('Target output example:',target_output_train[0])
        

        target_output_test = np.copy(decoder_input_test)
        
        target_output_test = removeSOS(target_output_test,len(test))
        
        print('Target output:',target_output_test.shape)
        print('Target output example:',target_output_test[0])
        print('\n\n')
                
        
                
                
        
        #make the target as an one hot encoding
        target_output_train = oneHotEncoding(target_output_train, output_vocabulary_size)


        #make the target as an one hot encoding
        target_output_test = oneHotEncoding(target_output_test, output_vocabulary_size)
        
                
 
        return encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, input_vocabulary_size, input_tokenizer, input_max_sentence_length, output_vocabulary_size, output_tokenizer, output_max_sentence_length, train, test
        
        
        
        
        
        
        
        
        
        
        
        
        
     
        
     
    
    
           
     
        
     
    
    
           
