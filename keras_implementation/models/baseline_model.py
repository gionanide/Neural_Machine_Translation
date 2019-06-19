#!usr/bin/python
import keras

#------------------------------------------------------------------------------------------------> NMT model
def NMT(input_vocabulary_size, output_vocabulary_size, input_max_sentence_length, output_max_sentence_length, n_units):


        #initalize properties
        print('\n')
        print('english_vocabulary_size ---->',output_vocabulary_size)
        print('german_vocabulary_size ---->',input_vocabulary_size)
        print('english_max_sentence_length ---->',output_max_sentence_length)
        print('german_max_sentence_length ---->',input_max_sentence_length)
        print('n_units ---->',n_units)
        print('\n')



        
        #input layer
        nmt_input_layer = keras.layers.Input(shape=(input_max_sentence_length,),name='nmt_input_layer')
        
        #embedding
        nmt_embedding_layer = keras.layers.Embedding(input_dim=input_vocabulary_size, output_dim=n_units, input_length=input_max_sentence_length, mask_zero=True, name='nmt_embedding_layer')
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------> ENCODER
        #lstm
        nmt_lstm_layer0 = keras.layers.LSTM(n_units, dropout=0, return_sequences=False, return_state=False, name='nmt_lstm_layer_0')
        
        #repeatvector
        nmt_repeat_vector = keras.layers.RepeatVector(output_max_sentence_length, name='nmt_repeat_vector')
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------> DECODER
        #lstm1
        nmt_lstm_layer1 = keras.layers.LSTM(n_units, dropout=0, return_sequences=True, return_state=False, name='nmt_lstm_layer_1')
        
        #timedistributed
        nmt_time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(output_vocabulary_size, activation='softmax'),name='nmt_time_distributed')
        
        
        
        #-----> sequence
        
        #input        
        print('Input layer output -------------------------->',nmt_input_layer.shape)
        
        #embedding
        embedding_output = nmt_embedding_layer(nmt_input_layer)
        
        print('Embedding output -------------------------->',embedding_output.shape)
        
        #lstm0
        # lstm0_output_hidden_sequence, lstm0_output_h, lstm0_output_c
        lstm0_output_hidden_sequence = nmt_lstm_layer0(embedding_output)
        
        print('LSTM-0 output -------------------------->',lstm0_output_hidden_sequence.shape)
        
        #repeat_vector
        repeat_vector_output = nmt_repeat_vector(lstm0_output_hidden_sequence)
        
        print('Repeat vector output -------------------------->',repeat_vector_output.shape)
        print(repeat_vector_output)
        
        #lstm1
        lstm1_output_hidden_sequence = nmt_lstm_layer1(repeat_vector_output)
        
        print('LSTM-1 output -------------------------->',lstm1_output_hidden_sequence.shape)
        
        #time distributes
        time_distributed_output = nmt_time_distributed(lstm1_output_hidden_sequence)
        
        print('Time distributed output -------------------------->',time_distributed_output.shape)
        
        
        full_model = keras.models.Model(inputs=[nmt_input_layer],outputs=[time_distributed_output])
        
        print(full_model.summary())
        
        #keras.utils.vis_utils.plot_model(full_model, show_shapes=True, show_layer_names=True, to_file='nmt_model.png')
        
        return full_model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
