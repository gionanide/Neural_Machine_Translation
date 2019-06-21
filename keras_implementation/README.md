#-------------------------------------------------------------> Bahdanau Attention layer

#use the layer as follows

import bahdanau_attention as bahdanau_attention

#input 
#(batch_size, timesteps, features_per_timestep)
#(batch_size, features_per_timestep)
 
bahdanau_attention.BahdanauAttention(n_units, name='bahdanau_attention_layer')

context_vector, attention_weights = bahdanau_attention_layer([hidden_state, hidden_sequence])

#output 
#context vector: (batch_size, features_per_timestep)
#attention weights: (batch_size, timesteps, 1)
