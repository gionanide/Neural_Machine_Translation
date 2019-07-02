#!usr/bin/python
import torch



#make a seperate class for Bahdanau attention layer
class BahdanauAttention(torch.nn.Module):

        #initialize the class calling the default function
        def __init__(self, units):
        
                #because we need subclass model
                super(BahdanauAttention, self).__init__()
        
                #make two dense layers because need a weighted sum
                self.W_1 = torch.nn.Linear(units, units)
                self.W_2 = torch.nn.Linear(units, units)      
                
                #define the output 
                self.V = torch.nn.Linear(units, 1)
                


        def forward(self, encoder_inputs, decoder_hidden):
        
                #print('\n\n\n')
                #print('--------------------- Start Bahdanau Attention ---------------------')
        
                #assign the variables
                #print('Bahnadau Attention encoder_inputs ---------------->',encoder_inputs.shape) #(hidden states, batch size, features)
                #print('Bahnadau Attention input2 ---------------->',decoder_hidden.shape) #(1, 1, features)

   
                
                #find the score for every hidden state, shape ----> (batch_size, max_length, hidden_size)
                #in our case the batch_size is the reviews so we assign weight to every review
                score = self.V(torch.tanh(self.W_1(decoder_hidden) + self.W_2(encoder_inputs)))                
                #print('Scores:',score.shape)
                
                
                #attention weights shape ----> (batch_size, max_length, 1), we conclude with 1 because we got the score back
                #-----> axis=0, iterate along the rows
                #-----> axis=1, iterate along the columns
                attention_weights = torch.softmax(score, dim=0)
                #print('Attention weights:',attention_weights.shape)
                
                #take the context vector
                context_vector = attention_weights * encoder_inputs
                #print('Context vector:',context_vector.shape)
                
                
                #one weight for every hidden state, output ----> (batch_size, hidden_size)
                context_vector = torch.sum(context_vector, dim=0).unsqueeze(0)
                #print('Context vector:',context_vector.shape)
                
                #the outputs the model returns, RECALL: the outputs must be returned as a list [output1, output2, ...... ,outputN]
                return context_vector, attention_weights
                
                
        


























