#!usr/bin/python
import keras
import sys
import tensorflow as tf

#insert the path to shared file first and then import the scripts
sys.path.insert(0, '/media/data/gionanide/shared_python_scripts')

import bahdanau_attention as bahdanau_attention
import gpu_initializations as gpu_init

#initialize some properties
sess = gpu_init.CUDA_init(core='GPU',memory='dynamically')



#------------------------------------------------------------------------------------------------> NMT model
def NMT(input_vocabulary_size, output_vocabulary_size, input_max_sentence_length, output_max_sentence_length, hidden_units, embedding_dimension, dropout):


        #initalize properties
        print('\n')
        print('------- Model properties -------')
        print('hidden_units ---->',hidden_units)
        print('embedding dimension ---->',embedding_dimension)
        print('dropout rate ---->',dropout)
        print('\n')


        #-------------------------------------------------------------------------------------------------------------------------------------------------------> ENCODER


        #----------------------------------------------------------------------------------> INPUT
        #input layer
        encoder_inputs = keras.layers.Input(shape=(input_max_sentence_length,),name='encoder_inputs') 
        #print('Encoder input -------------------------->',encoder_inputs.shape)
        
        
        #----------------------------------------------------------------------------------> EMBEDDING
        #embedding
        encoder_embedding = keras.layers.Embedding(input_dim=input_vocabulary_size, output_dim=embedding_dimension, input_length=input_max_sentence_length, mask_zero=True, name='encoder_embedding')
        #embedding
        encoder_embedding_output = encoder_embedding(encoder_inputs)
        #print('Encoder embedding output -------------------------->',encoder_embedding_output.shape)
 
 
 
        #----------------------------------------------------------------------------------> LSTM BIDIRECTIONAL
        #lstm_bidirectional
        encoder_lstm_layer0 = keras.layers.Bidirectional(keras.layers.LSTM(hidden_units, dropout=0, return_sequences=True, return_state=False, name='bidirectional'))
        lstm0_output_hidden_sequence = encoder_lstm_layer0(encoder_embedding_output)
        #print('Encoder LSTM-0 output -------------------------->',lstm0_output_hidden_sequence.shape)
        
        
        
        #----------------------------------------------------------------------------------> LSTM SUMMARIZATION
        #lstm for summarization
        encoder_lstm_layer01 = keras.layers.LSTM(hidden_units, dropout=dropout, return_sequences=False, return_state=True, name='summarization')
        lstm01_output_hidden_sequence, lstm01_output_h, lstm01_output_c = encoder_lstm_layer01(lstm0_output_hidden_sequence)
        #print('Encoder LSTM-01 output -------------------------->',lstm01_output_hidden_sequence.shape)
        #print(lstm01_output_hidden_sequence.type)
        

        #take encoder states to feed the decoder
        encoder_states = [lstm01_output_h, lstm01_output_c]

        
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------> DECODER        
        
        #input layer
        decoder_inputs = keras.layers.Input(shape=(output_max_sentence_length,),name='dencoder_inputs') 
        #print('Encoder input -------------------------->',decoder_inputs.shape)
        
        
        
        #----------------------------------------------------------------------------------> EMBEDDING
        #embedding
        dencoder_embedding = keras.layers.Embedding(input_dim=output_vocabulary_size, output_dim=embedding_dimension, input_length=output_max_sentence_length, mask_zero=True, name='dencoder_embedding')
        #embedding
        dencoder_embedding_output = dencoder_embedding(decoder_inputs)
        print('Dencoder embedding output -------------------------->',dencoder_embedding_output.shape)
        
        
        
        #repeatvector
        nmt_repeat_vector = keras.layers.RepeatVector(output_max_sentence_length, name='nmt_repeat_vector')
        lstm_context_vector = nmt_repeat_vector(lstm01_output_hidden_sequence)
        print('Repeat vector output -------------------------->',lstm_context_vector.shape)
        

        #----------------------------------------------------------------------------------------> CONCATENATE
        concat_encoder_decoder = keras.layers.Concatenate(axis=-1,name='concat')([lstm_context_vector, dencoder_embedding_output])
        print('Concatenate context vector with decoder input -------------------------->',concat_encoder_decoder.shape)
        
        
        decoder_lstm = keras.layers.LSTM(hidden_units, dropout=dropout, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_lstm_output, decoder_state_h, decoder_state_c = decoder_lstm(concat_encoder_decoder, initial_state=encoder_states)
        print('Decoder LSTM output -------------------------->',decoder_lstm_output.shape)
        
        
        decoder_dropout = keras.layers.Dropout(0.2)
        decoder_dropout_output = decoder_dropout(decoder_lstm_output)
        #print('Encoder Dropout output -------------------------->',decoder_dropout_output.shape)
                
                
                
        decoder_dense = keras.layers.Dense(output_vocabulary_size, activation='relu')
        decoder_outputs = decoder_dense(decoder_dropout_output)
        #print('Decoder outputs -------------------------->',decoder_outputs.shape)
        

        #make the model
        full_model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs],outputs=[decoder_outputs])
        
        #check the parameters and the layers details
        print(full_model.summary())
        
        #same the model's graph as an image
        #keras.utils.vis_utils.plot_model(full_model, show_shapes=True, show_layer_names=True, to_file='seq2seq.png')
        
        return full_model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
