Python scripts shared folder to initialize some global properties.


```python


#if you want to import from a file in a specific directory
sys.path.insert(0, '/media/data/gionanide/shared_python_scripts')


#-----------------------------------------------------------------------------> Tensorflow Session properties
#default tensorflow session allocates all the available memory

# Allocates memory dynamically, as much as you need
gpu_options = tf.GPUOptions(allow_growth=True)

# set a fraction ot the memory allocation
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


#choose which GPU  to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"
        
#user CPU
os.environ["CUDA_VISIBLE_DEVICES"]=""


#initialize tensorflow session
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)


#-----------------------------------------------------------------------------> Bahdanau Attention layer

#use the layer as follows

import bahdanau_attention as bahdanau_attention

#input 
#(batch_size, timesteps, features_per_timestep)
#(batch_size, features_per_timestep)
 
bahdanau_attention.BahdanauAttention(n_units, name='bahdanau_attention_layer')

#output 
#context vector: (batch_size, features_per_timestep)
#attention weights: (batch_size, timesteps, 1)



```
