#!usr/bin/python
#import tensorflow as tf
import keras
import numpy as np
import gpu_initializations as gpu_init
import numpy as np



#make a seperate class for Bahdanau attention layer
class BahdanauAttention(keras.Model):

        #initialize the class calling the default function
        def __init__(self, units, **kwargs):
        
                #because we need subclass model
                super(BahdanauAttention, self).__init__(**kwargs)
        
                #make two dense layers because need a weighted sum
                self.W_1 = keras.layers.Dense(units)
                self.W_2 = keras.layers.Dense(units)      
                
                #define the output 
                self.V = keras.layers.Dense(1)
                


        def call(self, call_input):
        
                print('\n\n\n')
                print('--------------------- Start Bahdanau Attention ---------------------')
        
                #assign the variables
                print('Bahnadau Attention input1 ---------------->',call_input[0])
                print('Bahnadau Attention input2 ---------------->',call_input[1])
                query = call_input[0]
                values = call_input[1]
        
        
                #the hidden shape is (batch_size, hidden size)
                #hidden with timesteps (batch_size, timesteps, hidden_size)
                hidden_with_timestep = keras.backend.expand_dims(query, 1) # transforms to [batch_size, 1, hidden_size]
                
                
                #find the score for every hidden state, shape ----> (batch_size, max_length, hidden_size)
                #in our case the batch_size is the reviews so we assign weight to every review
                score = self.V(keras.activations.tanh(self.W_1(values) + self.W_2(hidden_with_timestep)))
                
                #attention weights shape ----> (batch_size, max_length, 1), we conclude with 1 because we got the score back
                #-----> axis=0, iterate along the rows
                #-----> axis=1, iterate along the columns
                attention_weights = keras.activations.softmax(score, axis=1)
                
                #take the context vector
                context_vector = attention_weights * values
                
                
                #one weight for every hidden state, output ----> (batch_size, hidden_size)
                context_vector = keras.backend.sum(context_vector, axis=1)
                
                #the outputs the model returns, RECALL: the outputs must be returned as a list [output1, output2, ...... ,outputN]
                return [context_vector, attention_weights]
                
                
        #define the output shape of the model
        def compute_output_shape(self, input_shape):
        
                print('Bahnadau Attention input_shape1 (last_hidden_state_h)---------------->',input_shape[0])
                print('Bahnadau Attention input_shape2 (hidden_states_sequence)---------------->',input_shape[1])
                
                print('Bahnadau Attention output_shape_join ---------------->',[(input_shape[0][0], input_shape[0][1]),(input_shape[1][0], input_shape[1][1], 1)])
                
                
                print('--------------------- Start Bahdanau Attention ---------------------')
                print('\n\n\n')

                #return the output shape as a list of tuples, one tuple for every output, RECALL: output shapes must be [(), (), (), ...... ,()]
                return [(input_shape[0][0], input_shape[0][1]),(input_shape[1][0], input_shape[1][1], 1)]






'''
#sess = gpu_init.CUDA_init('GPU','dynamically')        
                

#lstm_output_sequence = tf.placeholder(tf.float32, shape=[12,5,300])
lstm_output_sequence = np.array([12,5,300])



#lstm_output_sequence = np.random.random((12,5,300)) #(batch_size, max_sentence, features for every sentence)
#lstm_output_hidden_state = tf.placeholder(tf.float32, shape=[12,300])

lstm_output_hidden_state = np.array([12,300])



#lstm_output_hidden_state = np.random.random((12,300)) #(batch_size, features for every sentence (hidden states))
print('lstm_output_hidden_state tensor:',lstm_output_sequence.shape)
print('lstm_output_sequence tensor:',lstm_output_hidden_state.shape)
                
                
                
attention_layer = BahdanauAttention(300)
                
attention_result, attention_weights = attention_layer.call(lstm_output_hidden_state, lstm_output_sequence)

print('\n\n')
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))

#print(attention_weights.eval())
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
'''
















































