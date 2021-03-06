## ------------------------------ baseline model ------------------------------

![alt text](https://github.com/gionanide/Neural_Machine_Translation/blob/master/keras_implementation/models/baseline_model.png)
```python

Layer (type)                 Output Shape              Param #   
=================================================================
nmt_input_layer (InputLayer) (None, 12)                0         
_________________________________________________________________
nmt_embedding_layer (Embeddi (None, 12, 256)           1904384   
_________________________________________________________________
nmt_lstm_layer_0 (LSTM)      (None, 256)               525312    
_________________________________________________________________
nmt_repeat_vector (RepeatVec (None, 6, 256)            0         
_________________________________________________________________
nmt_lstm_layer_1 (LSTM)      (None, 6, 256)            525312    
_________________________________________________________________
nmt_time_distributed (TimeDi (None, 6, 4681)           1203017   
=================================================================
Total params: 4.158.025
Trainable params: 4.158.025
Non-trainable params: 0


```


## ------------------------------ teacher forcing ------------------------------

![alt text](https://github.com/gionanide/Neural_Machine_Translation/blob/master/keras_implementation/models/teacher_forcing.png)
```python

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
encoder_inputs (InputLayer)     (None, 14)           0                                            
__________________________________________________________________________________________________
encoder_embedding (Embedding)   (None, 14, 256)      1904896     encoder_inputs[0][0]             
__________________________________________________________________________________________________
dencoder_inputs (InputLayer)    (None, 8)            0                                            
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 14, 512)      1050624     encoder_embedding[0][0]          
__________________________________________________________________________________________________
dencoder_embedding (Embedding)  (None, 8, 256)       1198848     dencoder_inputs[0][0]            
__________________________________________________________________________________________________
summarization (LSTM)            [(None, 14, 256), (N 787456      bidirectional_1[0][0]            
__________________________________________________________________________________________________
decoder_lstm (LSTM)             [(None, 8, 256), (No 525312      dencoder_embedding[0][0]         
                                                                 summarization[0][1]              
                                                                 summarization[0][2]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8, 256)       65792       decoder_lstm[0][0]               
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 8, 256)       0           dense_1[0][0]                    
__________________________________________________________________________________________________
decoder_dense (Dense)           (None, 8, 4683)      1203531     dropout_1[0][0]                  
==================================================================================================
Total params: 6,736,459
Trainable params: 6,736,459
Non-trainable params: 0

```
