Evaluation procedure, it is a seperate file because maybe you want to adjust it to your needs.

```python

'''

make a distinct function in order to compile and train the model

'''
def compile_train(model, encoder_input_train, target_output_train, encoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer, decay):     
def compile_train_teacher_forcing(model, encoder_input_train, decoder_input_train, target_output_train, encoder_input_test, decoder_input_test, target_output_test, loss, epochs, learning_rate, batch_size, dropout_lstm_encoder, dropout_lstm_decoder, dropout_layer, decay): 
'''

this functions makes a mapping from index to work

input: 1 (index)
output: sos (word)
   
'''   
#make the reverse mapping to go from indexing to the real word 
def index_to_word_mapping(requested_index, tokenizer):




'''

With this function we are taking the predictions that the model made and we are trying to decode them and make the sentence

------ inputs ------

- model: our trained model
- tokenizer: makes the mapping from word to index and the reverse one,   [1->sos]   and   [sos->1]
- input sequence, a matrix with dimensions  (max_sentence_length x size_of_vocabulary) recall the one hot encoding

------ output ------

- sentence

'''
#make this function in order to generate the predicted output in words
def generate_predicted_sequece(model, tokenizer, input_sequence):  




'''

BLEU example

two sentences: I want to check if the estimated one approached the ground_truth by comparing the n-grams


ground_truth = [['this','is','a','huge','banana']]
estimated = ['this','is','a','big','banana']

1-gram: (1, 0, 0, 0)
2-gram: (0.5, 0.5, 0, 0)
3-gram: (0.33, 0.33, 0.33, 0)
4-gram: (0.25, 0.25, 0.25, 0.25)

Cumulative 1-gram: 0.8
Cumulative 2-gram: 0.6324555320336759
Cumulative 3-gram: 0.5143157015215767
Cumulative 4-gram: 7.380245217279165e-78

'''
#make the BLUE metric as a function 
def BLUE_metric(ground_truth, estimated):




'''

Join all the previous functions to decode the input sequence, then provide the resulting sentence, and calculate the BLEU score. Overall validation calculate the mean BLEU score.

'''
#make this function in order to evaluate the model in speech generation based on metrics like BLUE, ROGUE etc
def model_speech_evaluation(model, tokenizer, input_sequences, dataset, role):        
'''

Same procedure as the above but formated for the teacher forcing procedure

'''     
#make this function in order to evaluate the model in speech generation based on metrics like BLUE, ROGUE etc
def model_speech_evaluation_teacher_forcing(model, tokenizer, input_sequences, dataset, role):

```
        
        
        
        
        
        
        
        
        
        
        
        
        
     
     
     
     
