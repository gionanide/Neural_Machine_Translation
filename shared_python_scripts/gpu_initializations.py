#!usr/bin/python
import os
import tensorflow as tf
import keras


'''

call this function to initialize the following properties

'''  
def CUDA_init(core,memory,parallel):

        print('\n ----------------- Start initalizing CUDA properties ----------------- \n')

        '''
        initialize some properties concerning CUDA, Peripheral Component Interconnect (PCI), a computer bus a system of communication that transfers data between components inside the computer. Recall that we user this command because the GPU ID that is used by CUDA and GPU ID used by non-CUDA programs like nvidia-smi are different. CUDA tries to associate the fastest GPU with the lowest ID. Non-CUDA tools use the PCI Bus ID of the GPUs to give them a GPU ID. With the following command, we request from CUDA to user the bus IDs of the GPUs instead of it'w own internal IDs.
        '''
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        
        
        if(core == "GPU"):
        
        
                if(parallel):
        
                        #running the example on multiple gpus
                        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
                        
                        print('\n ----------------- Using multiple GPUs ----------------- \n')
                        
                else:
        
                        #define which GPU to use, the number is the ID of the GPU given from your system (bus ID)
                        os.environ["CUDA_VISIBLE_DEVICES"]="0"
                
                
                        print('\n ----------------- Using GPU ----------------- \n')
                        
        
        elif(core == "CPU"):
        
                #run the following command if you want to force keras to run on CPU
                os.environ["CUDA_VISIBLE_DEVICES"]=""
                
                print('\n ----------------- Using CPU ----------------- \n')
                
        
        
        #use exactly the GPU percentage you actally need, because keras default procedure allocates all memory
        
        #-----> per_process_gpu_memory_fraction=0.333 ----- use this command if you want to fix the amount of GPU memory you want to allocate
        
        #-----> allow_growth=True ----- alternative use this command to dynamically use the amount of GPU memory that is necessary
        
        if(memory == 'dynamically'):
        
                # after this initialize the session
                gpu_options = tf.GPUOptions(allow_growth=True)
                
                print('\n ----------------- Dynamic memory allocation ----------------- \n')
                
                
        elif(memory == 'fraction'):
                
                #if you use fraction, you have to define the percentage of gpu to use
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
                
                print('\n ----------------- Fixed memory size allocated ----------------- \n')
                
                
                
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        keras.backend.tensorflow_backend.set_session(sess)
        print('\n ----------------- End initalizing CUDA properties ----------------- \n')
        print('\n\n\n')
        
        return sess
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
