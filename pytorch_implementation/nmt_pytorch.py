#!usr/bin/python
from __future__ import print_function, division
import encoder
import sys
#insert the path to shared file first and then import the scripts
sys.path.insert(0, '/media/data/gionanide/LSTMs')
import re
import string
from unicodedata import normalize
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import nltk
import preprocessing as preprocessing
import evaluate_model as evaluate_model
import encoder as encoder
import decoder as decoder
import bahdanau_attention_pytorch as bahdanau_attention
import torch
import evaluation as evaluation
from time import perf_counter as pc
import gpu_initialization_torch as gpu_init


#torch properties

#reproducibility
torch.manual_seed(0)

#select GPU
#device = gpu_init.CUDA_init_torch(core='GPU')

device = torch.device('cuda:0')


if __name__=='__main__':

        path = '/media/data/gionanide/LSTMs/deu.txt'
        
        text = preprocessing.readFile(path)
        
        pairs = preprocessing.makePairs(text)
        
        cleaned_words = preprocessing.cleaning(pairs,forcing=True)
        
        #visualize(cleaned_words)
        #print(cleaned_words.shape) --> (192881, 2) 0 -english || 1 -german
        
        
        cleaned_words = cleaned_words[:10000, :]
        
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
        train = cleaned_words[:9000]
        test = cleaned_words[9000:]
        
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
        
        
        
        
        
        
        #--------------------------------------------------------------------> Encoder inputs
        print('\n\n')
        print('Encoder format:  "sentence"<EOS>')
        encoder_inputs_train = np.copy(train_german_input)
        
        encoder_inputs_train = preprocessing.removeSOS(encoder_inputs_train,len(train))

        print('Encoder inputs train:',encoder_inputs_train.shape)
        print('Encoder inputs train example:',encoder_inputs_train[0])
        
        
        encoder_inputs_test = np.copy(test_german_input)
        
        encoder_inputs_test = preprocessing.removeSOS(encoder_inputs_test,len(test))
        
        print('Encoder inputs test:',encoder_inputs_test.shape)
        print('Encoder inputs test example:',encoder_inputs_test[0])
        print('\n\n')
        
        
        
        #-------------------------------------------------------------------------> Decoder inputs
        print('\n\n')
        print('Decoder format:  <SOS>"sentence"')
        decoder_inputs_train = np.copy(train_english_output)
        
        decoder_inputs_train = preprocessing.removeEOS(decoder_inputs_train)

  
        print('Decoder inputs train:',decoder_inputs_train.shape)
        print('Decoder input train example:',decoder_inputs_train[0])


        decoder_inputs_test = np.copy(test_english_output)
        
        decoder_inputs_test = preprocessing.removeEOS(decoder_inputs_test)
        

        print('Decoder inputs test:',decoder_inputs_test.shape)
        print('Decoder input example test:',decoder_inputs_test[0])
        print('\n\n')
        
        
        #----------------------------------------------------------------------------------------------------> Target outputs
        print('\n\n')
        print('Target format:  "sentence"<EOS>')
        target_outputs_train = np.copy(train_english_output)
        
        target_outputs_train = preprocessing.removeSOS(target_outputs_train,len(train))
        
        #target_outputs_train = preprocessing.oneHotEncoding(np.copy(target_outputs_train), english_vocabulary_size)

        print('Target output:',target_outputs_train.shape)
        print('Target output example:',target_outputs_train[0])
        

        target_outputs_test = np.copy(test_english_output)
        
        target_outputs_test = preprocessing.removeSOS(target_outputs_test,len(test))
        
        print('Target output:',target_outputs_test.shape)
        print('Target output example:',target_outputs_test[0])
        print('\n\n')
        
        #target_outputs_test = preprocessing.oneHotEncoding(np.copy(target_outputs_test), english_vocabulary_size)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #convert the encoder input to Pytorch Tensor
        encoder_inputs_train = torch.tensor(encoder_inputs_train, dtype=torch.long, device=device)
        
        encoder_inputs_test = torch.tensor(encoder_inputs_test, dtype=torch.long, device=device)
        
        #encoder_input = encoder_inputs_train[0].unsqueeze(1)
        
        #check the size of the tensor
        print('encoder input tensor:',encoder_inputs_train.size())
        
        #convert the decoder as well
        decoder_inputs_train = torch.tensor(decoder_inputs_train, dtype=torch.long, device=device)
        
        decoder_inputs_test = torch.tensor(decoder_inputs_test, dtype=torch.long, device=device)
        
        #decoder_input = decoder_inputs_train[0].unsqueeze(1)
        
        print('decoder input tensor:',decoder_inputs_train.size())
        
        #and the target 
        target_outputs_train = torch.tensor(target_outputs_train, dtype=torch.long, device=device)
        
        target_outputs_test = torch.tensor(target_outputs_test, dtype=torch.long, device=device)
        
        #targe_output = target_outputs_train[0].unsqueeze(1)
        
        print('target output tensor:',target_outputs_train.size())
        
        
        
        
        
        
        
        #-----------------------------------------------------------------------> Training
        
        
        
        
        
        #initialize the encoder
        encoder = encoder.Encoder(german_vocabulary_size, 256, 256, 0)
        
        #initalize the decoder
        decoder = decoder.Decoder(256, english_vocabulary_size, dropout_percent=0)
        
        #initialize attention
        attention = bahdanau_attention.BahdanauAttention(256)
        
        
        




        
        
        
        
        #---------------------------------------------------------------------------> Parallism to GPUs
        parallel = False
        if (parallel):
        
        
                encoder = torch.nn.DataParallel(encoder)
                
                decoder = torch.nn.DataParallel(decoder)
                
                attention = torch.nn.DataParallel(attention)
                
                
                
        #move them to GPU
        encoder  = encoder.to(device)
        
        decoder  = decoder.to(device)
        
        attention  = attention.to(device)
        
        
        
        #-----------------------> define the optimization procedure
        learning_rate = 0.01
        epochs = 40
        
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        
        
        criterion = torch.nn.CrossEntropyLoss()
        
        
        
        #-----------------------------------------------------------------> start counting time
        start_time = pc()
        
        for epoch in range(epochs):
        
        
                plot_losses = []
                print_loss_total = 0
                plot_loss_total = 0
                
                print_every = 1000
        
                for index in range(1, encoder_inputs_train.size(0)+1):
                
                
                
                        
 
                        #-----------------------------------------------------------------------> Remove padding
                        #move the tensor to GPU
                        encoder_input = encoder_inputs_train[index-1].unsqueeze(1)
                        #print(encoder_input.shape)
                        #remove the zeros from the padding procedure
                        encoder_input = encoder_input[encoder_input.squeeze().nonzero()].squeeze(1)
                        #print(encoder_input.shape)
                        
                        decoder_input = decoder_inputs_train[index-1].unsqueeze(1)
                        #print(decoder_input.shape)
                        decoder_input = decoder_input[decoder_input.squeeze().nonzero()].squeeze(1)
                        #print(decoder_input.shape)
                        
                        target = target_outputs_train[index-1]
                        #print(target.shape)
                        #print(target)
                        target = target[target.nonzero()].squeeze()
                        #print(target)
                        #print(target.shape)
                        #print('\n')


                        #break
                   
                   
                        
                #break

                

                        predictions, attentions, loss = evaluation.train(encoder, decoder, encoder_input, decoder_input, target, encoder_optimizer, decoder_optimizer, criterion)
                          
                
                        print_loss_total+=loss
                        plot_loss_total+=loss
                        
                        if (index % print_every == 0):
                        
                                avg_loss = print_loss_total / print_every
                                
                                print_loss_total = 0
                                
                                #print(predictions)
                                #print(predictions.shape)

                                #_, prediction = predictions.max(dim=1)
                                #print(prediction)   
                                #print(prediction.shape)
                                #print(target)                  
                                
                                plot_losses.append(avg_loss)
                                print('epoch:',epoch,'sample:',index,'loss:',avg_loss)
                                #,end='\r' if you want to erase the previous output line
                                #sys.stdout.flush()                       
              
                
                running_loss=0


        print('Finished Training')
        
        end_time = pc()
        
        print('CPU training time: ', end_time - start_time,'secs')  
        
        
        plt.figure()
        plt.plot(plot_losses)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('per 100 samples')
        plt.legend(['Train'], loc='upper right')
        plt.show()   
        
                        
                      
                        
                        
                        
                       


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
