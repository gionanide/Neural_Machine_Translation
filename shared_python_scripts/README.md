Python scripts shared folder to initialize some global properties.

## Cuda properties
## Bahdanau attention


```python

import sys
#if you want to import from a file in a specific directory
sys.path.insert(0, '/media/data/gionanide/shared_python_scripts')


#-------------------------------------------------------------> Tensorflow Session properties

import gpu_initializations as gpu_init

#and call the function as follows
core = #choose between 'GPU'/'CPU'
memory = #choose between 'dynamically'/'fractions'

sess = gpu_init.CUDA_init(core=core,memory=memory)


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



```
