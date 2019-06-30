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
parallel = #choose between True/False

sess = gpu_init.CUDA_init(core=core,memory=memory,parallel=parallel)


```





