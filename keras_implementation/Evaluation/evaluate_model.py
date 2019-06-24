#!usr/bin/python
import re
import string
from unicodedata import normalize
import numpy as np
import keras
import sys
import matplotlib.pyplot as plt
import nltk
import statistics



'''

make a distinct function in order to compile and train the model

'''
def compile_train(model, encoder_input_train, target_output_train, encoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, decay):



        #define optimizer
        optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

        #compile the model based on the previous defined properties
        model.compile(optimizer=optimizer, loss=loss)   
        
        #start training procedure
        history = model.fit(encoder_input_train, target_output_train, epochs=epochs, batch_size=batch_size, validation_data=(encoder_input_test, target_output_test), verbose=1)   
        
        
        #visualize the results
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model training procedure')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()   
        
        return model 
        
        
        
def compile_train_teacher_forcing(model, encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, decay):


        #define optimizer
        optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

        #compile the model based on the previous defined properties
        model.compile(optimizer=optimizer, loss=loss)   
        
        #start training procedure
        history = model.fit([encoder_input_train,decoder_input_train], target_output_train, epochs=epochs, batch_size=batch_size, validation_data=([encoder_input_test,decoder_input_test], target_output_test), verbose=1)   
        
        
        #visualize the results
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()   
        
        return model
   
   
   
   
#make the reverse mapping to go from indexing to the real word 
def index_to_word_mapping(requested_index, tokenizer):

        #iterate all the elements in the ['word'] = index, mapping
        for word, index in tokenizer.word_index.items():
        
                #if the requested index is in the dictionary, return the corresponding word
                if (requested_index == index):
                
                        return word
                        
        #if the requested index is not in the dictionary, return None
        return None

  
  
#make this function in order to generate the predicted output in words
def generate_predicted_sequece(model, tokenizer, input_sequence):

        #make the prediction for an input sequence, we define the input format previously
        prediction = model.predict(input_sequence, verbose=0)[0]
        
        #take the max (the values closer to 1, which is the max value, give us the best results, recall one hot encoding)
        requested_indeces = [np.argmax(vector) for vector in prediction]
        
        target_sequence = []
        
        #iterate all the predicted values
        for index in requested_indeces:
        
                #decrypt the predictions
                word = index_to_word_mapping(index, tokenizer)
                
                #if there is no prediction stop the generation
                if word is None:
                        
                       break
                       
                target_sequence.append(word)
                
                
        #return the list by joining the elements with gap between them to make a sentence
        return ' '.join(target_sequence)
    
    
    
    
#make the BLUE metric as a function 
def BLUE_metric(ground_truth, estimated):

        ground_truth = [ground_truth.split()]
        estimated = estimated.split()

        #print(ground_truth)
        #print(estimated)


        #weights=(1.0, 0, 0, 0, 0,)

        result = nltk.translate.bleu_score.sentence_bleu(ground_truth, estimated, weights=(1, 0, 0, 0))
 
        return result
        
        
        
 
#make this function in order to evaluate the model in speech generation based on metrics like BLUE, ROGUE etc
def model_speech_evaluation(model, tokenizer, input_sequences, dataset, role):

        bleu_metrics = []
        target_sequence_speech = []
        predicted_target_sequence_speech = []
        
        print('\n')
        print(role)
        
        #iterate all the input sequences, which are input to the model
        for index, input_sequence in enumerate(input_sequences):
        
                #make the appropriatte format
                input_sequence = input_sequence.reshape((1, input_sequence.shape[0]))
                
                #generate translation
                translation = generate_predicted_sequece(model, tokenizer, input_sequence)
                
                speech_target, speech_input = dataset[index]
                
                if (index<2):
                
                
                        print('---------------------- New sample ----------------------')
                        print('input ----------> ',speech_input)
                        print('target ----------> ',speech_target)
                        print('predicted ---------->',translation)
                        print('BLEU score ---------->',BLUE_metric(speech_target, translation))
                        print('\n')
                        
                bleu_metrics.append(BLUE_metric(speech_target, translation))
                        
                
                #append the results to the list to keep a record
                target_sequence_speech.append(speech_target.split())
                
                
                predicted_target_sequence_speech.append(translation.split())
                
                
        mean_bleu_metrics = statistics.mean(bleu_metrics)
        print('Mean BLEU metrics in',role,':',mean_bleu_metrics)
        #print('BlUE-1:',BLUE_metric(target_sequence_speech, predicted_target_sequence_speech)) 
        
        
        
        
        
        
        
        
        
#make this function in order to evaluate the model in speech generation based on metrics like BLUE, ROGUE etc
def model_speech_evaluation_teacher_forcing(model, tokenizer, input_sequences, dataset, role):


        bleu_metrics = []
        target_sequence_speech = []
        predicted_target_sequence_speech = []
        
        print('\n')
        print(role)
        
        #iterate all the input sequences, which are input to the model
        for index in range(input_sequences[0].shape[0]):
        
                #make the appropriatte format
                input_sequence = [input_sequences[0][index].reshape((1, input_sequences[0][index].shape[0])), input_sequences[1][index].reshape((1, input_sequences[1][index].shape[0]))]
                
                #generate translation
                translation = generate_predicted_sequece(model, tokenizer, input_sequence)
                
                speech_target, speech_input = dataset[index]
                
                if (index<2):
                
                
                        print('---------------------- New sample ----------------------')
                        print('input ----------> ',speech_input)
                        print('target ----------> ',speech_target)
                        print('predicted ---------->',translation)
                        print('BLEU score ---------->',BLUE_metric(speech_target, translation))
                        print('\n')
                        
                bleu_metrics.append(BLUE_metric(speech_target, translation))
                        
                
                #append the results to the list to keep a record
                target_sequence_speech.append(speech_target.split())
                
                
                predicted_target_sequence_speech.append(translation.split())
                
                
        mean_bleu_metrics = statistics.mean(bleu_metrics)
        print('Mean BLEU metrics in',role,':',mean_bleu_metrics)
        #print('BlUE-1:',BLUE_metric(target_sequence_speech, predicted_target_sequence_speech))
     
     
     
