#!usr/bin/python
import torch


'''

call this function to initialize the following properties

'''  
def CUDA_init_torch(core):

        print('\n ----------------- Start initalizing CUDA properties ----------------- \n')


        if (core == 'CPU'):
        
                device = torch.device('cpu')
                
                print('\n ----------------- Using CPU ----------------- \n')
                
                
        elif (core == 'GPU'):
        
                #select GPU
                gpus = torch.cuda.device_count()
                
                print('GPU devices found:',torch.cuda.device_count())
                
                for gpu in range(gpus):
                
                        print('torch_id:',gpu,'model:',torch.cuda.get_device_name(gpu))
                        
                      
                        
                gpu = input('Which gpu to use: ')
                
                while( (int(gpu) < 0) or (int(gpu) > gpus) ):                
                        
                        
                        gpu = input('Which gpu to use: ')
                        
                        
                device = torch.device('cuda:'+gpu)



		parallel = input('Use Data Parallelism over multiple GPUs  True/False: ')

		'''
			
			in your main code if parallel if true you have to append this command concerning your model: ----- model = torch.nn.DataParallel(model) -----

		'''
                
                
                
                print('\n ----------------- Using',device,'GPU ----------------- \n')
                
                
                
        print('\n ----------------- End initalizing CUDA properties ----------------- \n')
        print('\n\n\n')
        
        
        return device, parallel
                


