#!usr/bin/python
import numpy as np
import keras
import matplotlib.pyplot as plt
import preprocessing as preprocessing
import sys
from models import teacher_forcing_model
from models import baseline_model
import evaluation
import numpy as np
from models import seq2seq_attention
#insert the path to shared file first and then import the scripts
sys.path.insert(0, 'path/shared_python_scripts')
import gpu_initializations as gpu_init
  
     
np.random.seed(7)
     
        
'''

our model is a small model, the complexity can be increased using a larger number of examples, length of phrases and increasing the size of the vocabulary

'''             
if __name__ == '__main__':


        #define if you want to run it in GPU or CPU
        core = 'GPU'
        
        #define if you want dynamically memory allocation or a fixed size
        memory = 'dynamically'
        
        #define if you want to run in on multiple GPUs
        parallel = False
        
        #initialize some properties
        gpu_init.CUDA_init(core=core,memory=memory,parallel=parallel)


        #define the path of where to read the file
        path = 'deu.txt'
        
        #define the dataset length, maximum = 192881
        dataset_length = 100
        
        #define the percentage of training data
        train_per_cent = 0.9
        
        '''        
        translation flow
        
        1) english -- > german : flow = 'etg'
        
        2) german -- > english : flow = 'gte'
        
        3) enlglish/german -- > german/english : flow = 'egtge'
        '''
        flow = 'gte'
        
        
        #------------------------------> simple model
        encoder_input_train, target_output_train, encoder_input_test, target_output_test, input_vocabulary_size, input_tokenizer, input_max_sentence_length, output_vocabulary_size, output_tokenizer, output_max_sentence_length, train, test = preprocessing.format_data(path, dataset_length, train_per_cent, flow)
        
        
        #------------------------------> teacher forcing
        #encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, input_vocabulary_size, input_tokenizer, input_max_sentence_length, output_vocabulary_size, output_tokenizer, output_max_sentence_length, train, test = preprocessing.teacher_forcing_format(path, dataset_length, train_per_cent, flow)
        
        
        #------------------------------> seq2seq attention
        #encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, input_vocabulary_size, input_tokenizer, input_max_sentence_length, output_vocabulary_size, output_tokenizer, output_max_sentence_length, train, test = preprocessing.seq2seq_attention_format(path, dataset_length, train_per_cent, flow)

        


        
        #-----------------------------------------------------------> Set
        print('\n ----------------- Feeding sets ----------------- \n')
        print('\n ---------- Encoder ----------')
        print('Encoder inputs train shape:',encoder_input_train.shape)
        print('Encoder inputs test shape:',encoder_input_test.shape)
        print('Input max sentence length:',input_max_sentence_length)
        print('Input vocabulary size:',input_vocabulary_size)
        print('\n ---------- Decoder ----------')
        print('Decoder outputs train shape:',target_output_train.shape)
        print('Decoder outputs test shape:',target_output_test.shape)
        print('Output max sentence length:',output_max_sentence_length)
        print('Ouput vocabulary size:',output_vocabulary_size)
        print('\n')
        

        #initialize the model
        
        #hidden units
        hidden_units = 512
        
        #define the embedding dimension of the embedding layer
        embedding_dimension = 512
        
        #define the dropout rate
        dropout_layer = 0.4
        
        #dropout rate in encoder summarization lstm
        dropout_lstm_encoder = 0.2
        
        #dropout rate in decoder lstm
        dropout_lstm_decoder = 0.2
        
        
        #-----------------------------------------------------------------------------------------> Define the model
        
        #---------------------------> baseline model
        nmt_model = baseline_model.NMT(input_vocabulary_size, output_vocabulary_size, input_max_sentence_length, output_max_sentence_length, hidden_units, embedding_dimension)
        
        #---------------------------> teacher forcing model
        #nmt_model = teacher_forcing_model.NMT(input_vocabulary_size, output_vocabulary_size, input_max_sentence_length, output_max_sentence_length, hidden_units, embedding_dimension, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer)
        
        #-----------------------------------> using attention
        #nmt_model = seq2seq_attention.NMT(input_vocabulary_size, output_vocabulary_size, input_max_sentence_length, output_max_sentence_length, hidden_units, embedding_dimension, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer)
        
        
        #check if you chose to train your model on multiple GPUs, default 2 GPUs
        if (parallel):
        
                #pass the model so as to use parallelism with multiple GPUs
                nmt_model = keras.utils.multi_gpu_model(nmt_model, gpus=[0, 1])
        
        
        
        #-----------------------------------------------------------------------------------------> Train the model
        
        #define loss function
        loss = 'categorical_crossentropy'
        
        #time to see the whole dataset during training
        epochs = 100
        
        #learning_rate
        learning_rate = 0.001
        
        #decay, decreasing of learning rate through time
        decay = 0
        
        #sampes to split the dataset for one epoch
        batch_size = 256
        
        #simple model
        trained_model = evaluation.compile_train(nmt_model, encoder_input_train, target_output_train, encoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer, decay)
        
        #teacher forcing
        #trained_model = evaluation.compile_train_teacher_forcing(nmt_model, encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer, decay)
        
 
        
        
        #-----------------------------------------------------------------------------------------> Evaluate model
        
        #baseline model
        evaluation.model_speech_evaluation(trained_model, output_tokenizer, encoder_input_train, train, role='Train')
        evaluation.model_speech_evaluation(trained_model, output_tokenizer, encoder_input_test, test, role='Test')
        
        #teacher forcing
        #evaluation.model_speech_evaluation_teacher_forcing(trained_model, output_tokenizer, [encoder_input_train, decoder_input_train], train, role='Train')
        #evaluation.model_speech_evaluation_teacher_forcing(trained_model, output_tokenizer, [encoder_input_test, decoder_input_test], test, role='Test')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
