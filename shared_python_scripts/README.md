Python scripts shared folder to initialize some global properties.

## Cuda properties
## Bahdanau attention


```python

import sys
#if you want to import from a file in a specific directory
sys.path.insert(0, 'path_to_file/shared_python_scripts')


#-------------------------------------------------------------> Tensorflow Session properties

import gpu_initializations as gpu_init

#and call the function as follows
core = #choose between 'GPU'/'CPU'
memory = #choose between 'dynamically'/'fractions'
parallel = True/False

sess = gpu_init.CUDA_init(core=core,memory=memory,parallel=parallel)


```

## Bahdanau attention

```python

# inputs: hidden_sequence from an LSTM (return_sequence true), and one input a state to calculate hidden_state_h
#output1: context vector, given attention to every slice of hidden_sequence
#output2: attention weights, every weight corresponding to every slice of the sequece

attention = bahdanau_attention.BahdanauAttention(300)
context_vector, attention_weights = neighbourhood_attention([hidden_state_h, hidden_sequence])


```





