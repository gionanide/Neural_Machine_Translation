Python scripts shared folder to initialize some global properties.


```python

#default tensorflow session allocates all the available memory

# Allocates memory dynamically, as much as you need
gpu_options = tf.GPUOptions(allow_growth=True)

# set a fraction ot the memory allocation
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


#choose which GPU  to choose
os.environ["CUDA_VISIBLE_DEVICES"]="0"
        
#user CPU
os.environ["CUDA_VISIBLE_DEVICES"]=""


#initialize tensorflow session
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)


```
