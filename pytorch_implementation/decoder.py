#!usr/bin/python
import torch
import bahdanau_attention_pytorch as bahdanau_attention


class Decoder(torch.nn.Module):


        def __init__(self, hidden_units, output_size, dropout_percent):
        
                super(Decoder, self).__init__()

                #initialize the properties
                self.hidden_units = hidden_units

                #range of Linear layer at the output
                self.output_size = output_size
                
                self.dropout_precent = dropout_percent
                
                
                #initialize layers

                #because we are using teacher forcing, we embed the input in the decoder (which is 1 timestep before the target)
                self.embedding = torch.nn.Embedding(self.output_size, hidden_units)
                
                
                self.dropout = torch.nn.Dropout(self.dropout_precent)
                
                #-----------------------------------------------------------------> Attention
                #make two dense layers because need a weighted sum
                self.W_1 = torch.nn.Linear(hidden_units, hidden_units)
                self.W_2 = torch.nn.Linear(hidden_units, hidden_units)      
                
                #define the output 
                self.V = torch.nn.Linear(hidden_units, 1)
                #-----------------------------------------------------------------> Attention


                self.lstm = torch.nn.LSTM(hidden_units, hidden_units)
                
                
                self.output = torch.nn.Linear(hidden_units, output_size)
                
                
                self.attn_combine = torch.nn.Linear(self.hidden_units * 2, self.hidden_units)


                self.softmax = torch.nn.LogSoftmax(dim=1)
                
                
                
                
        def forward(self, inputs, hidden_states, encoder_output_sequence):


                #print('--------------------------------- Decoder ---------------------------------')
                
                #---------------------------------------------------> Input
                #print('decoder input:',inputs.size())

                #-------------------------------------------> Embedding
                embedded = self.embedding(inputs)
                #print('embedded:',embedded.size())

                #----------------------------------------------> Relu activation
                output = self.dropout(embedded)
                #print('relu:',output.size())
                
                
                
                #-------------------------------------------------------------------------------------------------------------------------------> Attention
                #find the score for every hidden state, shape ----> (batch_size, max_length, hidden_size)
                #in our case the batch_size is the reviews so we assign weight to every review
                score = self.V(torch.tanh(self.W_1(output) + self.W_2(encoder_output_sequence)))                
                #print('Scores:',score.shape)
                
                
                #attention weights shape ----> (batch_size, max_length, 1), we conclude with 1 because we got the score back
                #-----> axis=0, iterate along the rows
                #-----> axis=1, iterate along the columns
                attention_weights = torch.softmax(score, dim=0)
                #print('Attention weights:',attention_weights.shape)
                
                #take the context vector
                context_vector = attention_weights * encoder_output_sequence
                #print('Context vector:',context_vector.shape)
                
                
                #one weight for every hidden state, output ----> (batch_size, hidden_size)
                context_vector = torch.sum(context_vector, dim=0).unsqueeze(0)
                #-------------------------------------------------------------------------------------------------------------------------------> Attention


                #-----------------------------------------------------------> LSTM 
                output, hidden = self.lstm(context_vector, hidden_states)
                #print('output:',output.size())
                #print(output[0].shape)

                #----------------------------------------------------------------------> Output
                output = torch.nn.functional.log_softmax(self.output(output[0]), dim=1)
                #print('softmax:',output.size())


                return output, hidden, attention_weights



        #def initHidden(self):


         #       return torch.zeros(1, 1, self.hidden_units, device=device)
                
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
